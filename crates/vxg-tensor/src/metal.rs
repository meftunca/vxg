//! Safe RAII Apple Metal GPU `GEMV` engine with persistent Unified Memory weight residency
//! (`0` per-token host-to-GPU weight copy). Compiles cleanly on non-macOS targets as a stub.

use crate::format::PackedTensor;
#[cfg(not(vxt_has_metal))]
use crate::Error;
use crate::{DType, Result};

#[cfg(vxt_has_metal)]
mod imp {
    use core::ffi::{c_char, c_void};
    use std::ffi::CStr;
    use std::ptr::NonNull;

    use crate::codebook::{CodebookMatrix, Q4_0Matrix};
    use crate::format::PackedTensor;
    use crate::fp16::fit_cubic_codebook;
    use crate::lossless::LosslessMatrix;
    use crate::{DType, Error, Result};

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

        fn vxt_metal_run_raw16(
            ctx: *mut c_void,
            w_buf: *mut c_void,
            x_ptr: *const f32,
            cols: u32,
            y_ptr: *mut f32,
            rows: u32,
            is_fp16: u32,
            iters: u32,
            out_gpu_ms: *mut f64,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> i32;

        fn vxt_metal_run_lossless(
            ctx: *mut c_void,
            base_h: *mut c_void,
            cw_h: *mut c_void,
            sb_h: *mut c_void,
            esc_h: *mut c_void,
            x_pad_ptr: *const f32,
            padded_cols: u32,
            y_ptr: *mut f32,
            rows: u32,
            is_fp16: u32,
            lut8: *const u32,
            iters: u32,
            out_gpu_ms: *mut f64,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> i32;

        fn vxt_metal_run_codebook(
            ctx: *mut c_void,
            sc_h: *mut c_void,
            pw_h: *mut c_void,
            row_ptr_h: *mut c_void,
            out_h: *mut c_void,
            x_pad_ptr: *const f32,
            padded_cols: u32,
            y_ptr: *mut f32,
            rows: u32,
            bits: u32,
            centroids16: *const f32,
            iters: u32,
            out_gpu_ms: *mut f64,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> i32;

        fn vxt_metal_run_q4_0(
            ctx: *mut c_void,
            sc_h: *mut c_void,
            qs_h: *mut c_void,
            x_pad_ptr: *const f32,
            padded_cols: u32,
            y_ptr: *mut f32,
            rows: u32,
            iters: u32,
            out_gpu_ms: *mut f64,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> i32;
    }

    fn cbuf_to_str(buf: &[u8]) -> String {
        let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        CStr::from_bytes_with_nul(&buf[..=(nul.min(buf.len().saturating_sub(1)))])
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|_| String::from_utf8_lossy(&buf[..nul]).into_owned())
    }

    pub struct GpuBuf(NonNull<c_void>);

    unsafe impl Send for GpuBuf {}
    unsafe impl Sync for GpuBuf {}

    impl GpuBuf {
        fn upload<T: Copy>(ctx: NonNull<c_void>, slice: &[T]) -> Result<Self> {
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
        fn as_ptr(&self) -> *mut c_void {
            self.0.as_ptr()
        }
    }

    impl Drop for GpuBuf {
        fn drop(&mut self) {
            unsafe { vxt_metal_free_buffer(self.0.as_ptr()) };
        }
    }

    pub struct MetalEngineInner {
        ctx: NonNull<c_void>,
        device_name: String,
        x_pad_scratch: Vec<f32>,
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
            })
        }

        pub fn device_name(&self) -> &str {
            &self.device_name
        }

        fn prepare_x_pad(&mut self, x: &[f32], padded_cols: usize) -> *const f32 {
            if self.x_pad_scratch.len() < padded_cols {
                self.x_pad_scratch.resize(padded_cols, 0.0);
            }
            self.x_pad_scratch[..x.len()].copy_from_slice(x);
            self.x_pad_scratch[x.len()..padded_cols].fill(0.0);
            self.x_pad_scratch.as_ptr()
        }
    }

    impl Drop for MetalEngineInner {
        fn drop(&mut self) {
            unsafe { vxt_metal_destroy(self.ctx.as_ptr()) };
        }
    }

    pub struct GpuRaw16Inner {
        pub dtype: DType,
        pub rows: usize,
        pub cols: usize,
        w_buf: GpuBuf,
    }

    impl GpuRaw16Inner {
        pub fn upload(
            eng: &MetalEngineInner,
            dtype: DType,
            rows: usize,
            cols: usize,
            words: &[u16],
        ) -> Result<Self> {
            if words.len() != rows.saturating_mul(cols) || rows == 0 || cols == 0 {
                return Err(Error::InvalidShape("invalid rows*cols for GpuRaw16Matrix"));
            }
            Ok(Self {
                dtype,
                rows,
                cols,
                w_buf: GpuBuf::upload(eng.ctx, words)?,
            })
        }

        pub fn run(
            &self,
            eng: &mut MetalEngineInner,
            x: &[f32],
            y: &mut [f32],
            iters: u32,
        ) -> Result<f64> {
            assert_eq!(x.len(), self.cols);
            assert_eq!(y.len(), self.rows);
            let runs = iters.max(1);
            let mut total_ms = 0.0f64;
            let mut err_buf = [0u8; 512];
            let rc = unsafe {
                vxt_metal_run_raw16(
                    eng.ctx.as_ptr(),
                    self.w_buf.as_ptr(),
                    x.as_ptr(),
                    self.cols as u32,
                    y.as_mut_ptr(),
                    self.rows as u32,
                    matches!(self.dtype, DType::F16) as u32,
                    runs,
                    &mut total_ms,
                    err_buf.as_mut_ptr() as *mut c_char,
                    err_buf.len(),
                )
            };
            if rc != 0 {
                return Err(Error::Backend(cbuf_to_str(&err_buf)));
            }
            Ok(total_ms / (runs as f64))
        }
    }

    pub enum GpuPackedInner {
        Lossless {
            dtype: DType,
            rows: usize,
            cols: usize,
            padded_cols: usize,
            lut8: [u32; 8],
            base: GpuBuf,
            cw: GpuBuf,
            sb: GpuBuf,
            esc: GpuBuf,
        },
        Codebook {
            bits: u8,
            rows: usize,
            cols: usize,
            padded_cols: usize,
            centroids: [f32; 16],
            sc: GpuBuf,
            pw: GpuBuf,
            row_ptr: GpuBuf,
            out: GpuBuf,
        },
        Q4_0 {
            rows: usize,
            cols: usize,
            padded_cols: usize,
            sc: GpuBuf,
            qs: GpuBuf,
        },
    }

    impl GpuPackedInner {
        pub fn upload(eng: &MetalEngineInner, tensor: &PackedTensor) -> Result<Self> {
            match tensor {
                PackedTensor::Lossless(m) => Self::upload_lossless(eng, m),
                PackedTensor::Codebook(m) => Self::upload_codebook(eng, m),
                PackedTensor::Q4_0(m) => Self::upload_q4_0(eng, m),
            }
        }

        fn upload_lossless(eng: &MetalEngineInner, m: &LosslessMatrix) -> Result<Self> {
            let mut lut8 = m.lut.map(|v| v as u32);
            lut8[7] = 0;
            Ok(Self::Lossless {
                dtype: m.dtype,
                rows: m.rows,
                cols: m.cols,
                padded_cols: m.padded_cols,
                lut8,
                base: GpuBuf::upload(eng.ctx, &m.base_bytes)?,
                cw: GpuBuf::upload(eng.ctx, &m.code_words)?,
                sb: GpuBuf::upload(eng.ctx, &m.sb_esc_offsets)?,
                esc: GpuBuf::upload(eng.ctx, &m.esc_bytes)?,
            })
        }

        fn upload_codebook(eng: &MetalEngineInner, m: &CodebookMatrix) -> Result<Self> {
            let k = 1usize << m.bits;
            let (poly, _) = fit_cubic_codebook(&m.centroids[..k]);
            let mut packed_params = [0.0f32; 16];
            packed_params[..4].copy_from_slice(&poly);
            Ok(Self::Codebook {
                bits: m.bits,
                rows: m.rows,
                cols: m.cols,
                padded_cols: m.padded_cols,
                centroids: packed_params,
                sc: GpuBuf::upload(eng.ctx, &m.scales_f16)?,
                pw: GpuBuf::upload(eng.ctx, &m.packed_words)?,
                row_ptr: GpuBuf::upload(eng.ctx, &m.outlier_row_ptr)?,
                out: GpuBuf::upload(eng.ctx, &m.outliers_packed)?,
            })
        }

        fn upload_q4_0(eng: &MetalEngineInner, m: &Q4_0Matrix) -> Result<Self> {
            Ok(Self::Q4_0 {
                rows: m.rows,
                cols: m.cols,
                padded_cols: m.padded_cols,
                sc: GpuBuf::upload(eng.ctx, &m.scales_f16)?,
                qs: GpuBuf::upload(eng.ctx, &m.qs_words)?,
            })
        }

        pub fn run(
            &self,
            eng: &mut MetalEngineInner,
            x: &[f32],
            y: &mut [f32],
            iters: u32,
        ) -> Result<f64> {
            let runs = iters.max(1);
            let mut total_ms = 0.0f64;
            let mut err_buf = [0u8; 512];

            let rc = match self {
                Self::Lossless {
                    dtype,
                    rows,
                    cols,
                    padded_cols,
                    lut8,
                    base,
                    cw,
                    sb,
                    esc,
                } => {
                    assert_eq!(x.len(), *cols);
                    assert_eq!(y.len(), *rows);
                    let xp = eng.prepare_x_pad(x, *padded_cols);
                    unsafe {
                        vxt_metal_run_lossless(
                            eng.ctx.as_ptr(),
                            base.as_ptr(),
                            cw.as_ptr(),
                            sb.as_ptr(),
                            esc.as_ptr(),
                            xp,
                            *padded_cols as u32,
                            y.as_mut_ptr(),
                            *rows as u32,
                            matches!(dtype, DType::F16) as u32,
                            lut8.as_ptr(),
                            runs,
                            &mut total_ms,
                            err_buf.as_mut_ptr() as *mut c_char,
                            err_buf.len(),
                        )
                    }
                }
                Self::Codebook {
                    bits,
                    rows,
                    cols,
                    padded_cols,
                    centroids,
                    sc,
                    pw,
                    row_ptr,
                    out,
                } => {
                    assert_eq!(x.len(), *cols);
                    assert_eq!(y.len(), *rows);
                    let xp = eng.prepare_x_pad(x, *padded_cols);
                    unsafe {
                        vxt_metal_run_codebook(
                            eng.ctx.as_ptr(),
                            sc.as_ptr(),
                            pw.as_ptr(),
                            row_ptr.as_ptr(),
                            out.as_ptr(),
                            xp,
                            *padded_cols as u32,
                            y.as_mut_ptr(),
                            *rows as u32,
                            *bits as u32,
                            centroids.as_ptr(),
                            runs,
                            &mut total_ms,
                            err_buf.as_mut_ptr() as *mut c_char,
                            err_buf.len(),
                        )
                    }
                }
                Self::Q4_0 {
                    rows,
                    cols,
                    padded_cols,
                    sc,
                    qs,
                } => {
                    assert_eq!(x.len(), *cols);
                    assert_eq!(y.len(), *rows);
                    let xp = eng.prepare_x_pad(x, *padded_cols);
                    unsafe {
                        vxt_metal_run_q4_0(
                            eng.ctx.as_ptr(),
                            sc.as_ptr(),
                            qs.as_ptr(),
                            xp,
                            *padded_cols as u32,
                            y.as_mut_ptr(),
                            *rows as u32,
                            runs,
                            &mut total_ms,
                            err_buf.as_mut_ptr() as *mut c_char,
                            err_buf.len(),
                        )
                    }
                }
            };

            if rc != 0 {
                return Err(Error::Backend(cbuf_to_str(&err_buf)));
            }
            Ok(total_ms / (runs as f64))
        }
    }
}

/// Apple Metal GPU compute engine for fused compressed-domain `GEMV`.
pub struct MetalEngine {
    #[cfg(vxt_has_metal)]
    inner: imp::MetalEngineInner,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

impl MetalEngine {
    pub fn is_compiled() -> bool {
        cfg!(vxt_has_metal)
    }

    pub fn new() -> Result<Self> {
        #[cfg(vxt_has_metal)]
        {
            Ok(Self {
                inner: imp::MetalEngineInner::new()?,
            })
        }
        #[cfg(not(vxt_has_metal))]
        {
            Err(Error::Unsupported(
                "Apple Metal GPU backend is only compiled on macOS".into(),
            ))
        }
    }

    pub fn device_name(&self) -> &str {
        #[cfg(vxt_has_metal)]
        {
            self.inner.device_name()
        }
        #[cfg(not(vxt_has_metal))]
        {
            "unavailable"
        }
    }

    pub fn upload_raw16(
        &self,
        dtype: DType,
        rows: usize,
        cols: usize,
        words: &[u16],
    ) -> Result<GpuRaw16Matrix> {
        #[cfg(vxt_has_metal)]
        {
            Ok(GpuRaw16Matrix {
                inner: imp::GpuRaw16Inner::upload(&self.inner, dtype, rows, cols, words)?,
            })
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (dtype, rows, cols, words);
            Err(Error::Unsupported("Metal backend unavailable".into()))
        }
    }

    pub fn upload_packed(&self, tensor: &PackedTensor) -> Result<GpuPackedTensor> {
        #[cfg(vxt_has_metal)]
        {
            Ok(GpuPackedTensor {
                inner: imp::GpuPackedInner::upload(&self.inner, tensor)?,
            })
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = tensor;
            Err(Error::Unsupported("Metal backend unavailable".into()))
        }
    }
}

/// Persistent Unified-Memory uncompressed 16-bit (`BF16`/`FP16`) matrix on Apple Metal GPU.
pub struct GpuRaw16Matrix {
    #[cfg(vxt_has_metal)]
    inner: imp::GpuRaw16Inner,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

impl GpuRaw16Matrix {
    /// Runs `iters` GPU `GEMV` dispatches (`y = W · x`) and returns average GPU kernel ms/iter.
    pub fn run_gemv(
        &self,
        engine: &mut MetalEngine,
        x: &[f32],
        y: &mut [f32],
        iters: u32,
    ) -> Result<f64> {
        #[cfg(vxt_has_metal)]
        {
            self.inner.run(&mut engine.inner, x, y, iters)
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (engine, x, y, iters);
            Err(Error::Unsupported("Metal backend unavailable".into()))
        }
    }
}

/// Persistent Unified-Memory compressed `.vxt` matrix (`Lossless`, `Cb4`, `Cb3`, or `Q4_0`) on GPU.
pub struct GpuPackedTensor {
    #[cfg(vxt_has_metal)]
    inner: imp::GpuPackedInner,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

impl GpuPackedTensor {
    /// Runs `iters` fused compressed-domain GPU `GEMV` dispatches and returns average GPU ms/iter.
    pub fn run_gemv(
        &self,
        engine: &mut MetalEngine,
        x: &[f32],
        y: &mut [f32],
        iters: u32,
    ) -> Result<f64> {
        #[cfg(vxt_has_metal)]
        {
            self.inner.run(&mut engine.inner, x, y, iters)
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (engine, x, y, iters);
            Err(Error::Unsupported("Metal backend unavailable".into()))
        }
    }
}
