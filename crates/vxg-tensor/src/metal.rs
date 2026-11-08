//! Safe RAII Apple Metal GPU `GEMV` engine with persistent Unified Memory weight residency
//! (`0` per-token host-to-GPU weight copy) and multi-matrix fused command-buffer execution.

use crate::format::PackedTensor;
#[cfg(not(vxt_has_metal))]
use crate::Error;
use crate::{DType, Result};

#[cfg(vxt_has_metal)]
mod imp {
    use std::ptr::null_mut;

    use crate::codebook::{CodebookMatrix, Q4_0Matrix};
    use crate::format::PackedTensor;
    use crate::fp16::fit_cubic_codebook;
    use crate::lossless::LosslessMatrix;
    pub use crate::metal_ffi::{GpuBuf, MetalEngineInner, VxtMetalOp};
    use crate::wht2::Wht2Matrix;
    use crate::wht_turbo::TurboWhtMatrix;
    use crate::{DType, Error, Result};

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
            eng.ops_scratch.clear();
            eng.ops_scratch.push(VxtMetalOp {
                kind: 0,
                sub: matches!(self.dtype, DType::F16) as u32,
                rows: self.rows as u32,
                cols: self.cols as u32,
                lut8: [0; 8],
                poly16: [0.0; 16],
                b0: self.w_buf.as_ptr(),
                b1: null_mut(),
                b2: null_mut(),
                b3: null_mut(),
                y_dst: y.as_mut_ptr(),
            });
            eng.dispatch_ops(x, self.cols, iters)
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
            poly16: [f32; 16],
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
        Wht2 {
            rows: usize,
            cols: usize,
            padded_cols: usize,
            poly16: [f32; 16],
            sc: GpuBuf,
            pw: GpuBuf,
            row_ptr: GpuBuf,
            out: GpuBuf,
        },
        RawF32 {
            rows: usize,
            cols: usize,
            w_buf: GpuBuf,
        },
        TurboWht {
            bits: u8,
            rows: usize,
            cols: usize,
            padded_cols: usize,
            sc: GpuBuf,
            pw: GpuBuf,
        },
    }

    impl GpuPackedInner {
        pub fn upload(eng: &MetalEngineInner, tensor: &PackedTensor) -> Result<Self> {
            match tensor {
                PackedTensor::Lossless(m) => Self::upload_lossless(eng, m),
                PackedTensor::Codebook(m) => Self::upload_codebook(eng, m),
                PackedTensor::Q4_0(m) => Self::upload_q4_0(eng, m),
                PackedTensor::Wht2(m) => Self::upload_wht2(eng, m),
                PackedTensor::RawF32 { rows, cols, data } => Ok(Self::RawF32 {
                    rows: *rows,
                    cols: *cols,
                    w_buf: GpuBuf::upload(eng.ctx, data)?,
                }),
                PackedTensor::TurboWht(m) => Self::upload_turbo(eng, m),
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
            let (poly, _) = fit_cubic_codebook(&m.centroids[..(1usize << m.bits)]);
            let mut poly16 = [0.0f32; 16];
            poly16[..4].copy_from_slice(&poly);
            Ok(Self::Codebook {
                bits: m.bits,
                rows: m.rows,
                cols: m.cols,
                padded_cols: m.padded_cols,
                poly16,
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

        fn upload_wht2(eng: &MetalEngineInner, m: &Wht2Matrix) -> Result<Self> {
            let (poly, _) = fit_cubic_codebook(&m.centroids[..4]);
            let mut poly16 = [0.0f32; 16];
            poly16[..4].copy_from_slice(&poly);
            Ok(Self::Wht2 {
                rows: m.rows,
                cols: m.cols,
                padded_cols: m.padded_cols,
                poly16,
                sc: GpuBuf::upload(eng.ctx, &m.scales_f16)?,
                pw: GpuBuf::upload(eng.ctx, &m.packed_words)?,
                row_ptr: GpuBuf::upload(eng.ctx, &m.outlier_row_ptr)?,
                out: GpuBuf::upload(eng.ctx, &m.outliers_packed)?,
            })
        }

        fn upload_turbo(eng: &MetalEngineInner, m: &TurboWhtMatrix) -> Result<Self> {
            Ok(Self::TurboWht {
                bits: m.bits,
                rows: m.rows,
                cols: m.cols,
                padded_cols: m.padded_cols,
                sc: GpuBuf::upload(eng.ctx, &m.scales_f16)?,
                pw: GpuBuf::upload(eng.ctx, &m.packed_words)?,
            })
        }

        pub fn raw_op(&self, y_dst: *mut f32) -> (VxtMetalOp, usize, usize, usize) {
            match self {
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
                } => (
                    VxtMetalOp {
                        kind: 1,
                        sub: matches!(dtype, DType::F16) as u32,
                        rows: *rows as u32,
                        cols: *padded_cols as u32,
                        lut8: *lut8,
                        poly16: [0.0; 16],
                        b0: base.as_ptr(),
                        b1: cw.as_ptr(),
                        b2: sb.as_ptr(),
                        b3: esc.as_ptr(),
                        y_dst,
                    },
                    *rows,
                    *cols,
                    *padded_cols,
                ),
                Self::Codebook {
                    bits,
                    rows,
                    cols,
                    padded_cols,
                    poly16,
                    sc,
                    pw,
                    row_ptr,
                    out,
                } => (
                    VxtMetalOp {
                        kind: 2,
                        sub: *bits as u32,
                        rows: *rows as u32,
                        cols: *padded_cols as u32,
                        lut8: [0; 8],
                        poly16: *poly16,
                        b0: sc.as_ptr(),
                        b1: pw.as_ptr(),
                        b2: row_ptr.as_ptr(),
                        b3: out.as_ptr(),
                        y_dst,
                    },
                    *rows,
                    *cols,
                    *padded_cols,
                ),
                Self::Q4_0 {
                    rows,
                    cols,
                    padded_cols,
                    sc,
                    qs,
                } => (
                    VxtMetalOp {
                        kind: 3,
                        sub: 0,
                        rows: *rows as u32,
                        cols: *padded_cols as u32,
                        lut8: [0; 8],
                        poly16: [0.0; 16],
                        b0: sc.as_ptr(),
                        b1: qs.as_ptr(),
                        b2: null_mut(),
                        b3: null_mut(),
                        y_dst,
                    },
                    *rows,
                    *cols,
                    *padded_cols,
                ),
                Self::Wht2 {
                    rows,
                    cols,
                    padded_cols,
                    poly16,
                    sc,
                    pw,
                    row_ptr,
                    out,
                } => (
                    VxtMetalOp {
                        kind: 4,
                        sub: 2,
                        rows: *rows as u32,
                        cols: *padded_cols as u32,
                        lut8: [0; 8],
                        poly16: *poly16,
                        b0: sc.as_ptr(),
                        b1: pw.as_ptr(),
                        b2: row_ptr.as_ptr(),
                        b3: out.as_ptr(),
                        y_dst,
                    },
                    *rows,
                    *cols,
                    *padded_cols,
                ),
                Self::RawF32 { rows, cols, w_buf } => (
                    VxtMetalOp {
                        kind: 5,
                        sub: 0,
                        rows: *rows as u32,
                        cols: *cols as u32,
                        lut8: [0; 8],
                        poly16: [0.0; 16],
                        b0: w_buf.as_ptr(),
                        b1: null_mut(),
                        b2: null_mut(),
                        b3: null_mut(),
                        y_dst,
                    },
                    *rows,
                    *cols,
                    *cols,
                ),
                Self::TurboWht {
                    bits,
                    rows,
                    cols,
                    padded_cols,
                    sc,
                    pw,
                } => (
                    VxtMetalOp {
                        kind: 6,
                        sub: *bits as u32,
                        rows: *rows as u32,
                        cols: *padded_cols as u32,
                        lut8: [*cols as u32, 0, 0, 0, 0, 0, 0, 0],
                        poly16: [0.0; 16],
                        b0: sc.as_ptr(),
                        b1: pw.as_ptr(),
                        b2: null_mut(),
                        b3: null_mut(),
                        y_dst,
                    },
                    *rows,
                    *cols,
                    *padded_cols,
                ),
            }
        }

        pub fn to_op(&self, x_len: usize, y: &mut [f32]) -> (VxtMetalOp, usize) {
            let (op, rows, cols, pcols) = self.raw_op(y.as_mut_ptr());
            assert_eq!((x_len, y.len()), (cols, rows));
            (op, pcols)
        }

        pub fn run(
            &self,
            eng: &mut MetalEngineInner,
            x: &[f32],
            y: &mut [f32],
            iters: u32,
        ) -> Result<f64> {
            eng.ops_scratch.clear();
            let (op, pcols) = self.to_op(x.len(), y);
            eng.ops_scratch.push(op);
            eng.dispatch_ops(x, pcols, iters)
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

    /// Executes multiple compressed `GEMV`s sharing the same input `x` inside ONE `MTLCommandBuffer`.
    pub fn run_multi_same_x(
        &mut self,
        x: &[f32],
        jobs: &mut [(&GpuPackedTensor, &mut [f32])],
    ) -> Result<f64> {
        #[cfg(vxt_has_metal)]
        {
            self.inner.ops_scratch.clear();
            let mut max_pcols = x.len();
            for (t, y) in jobs.iter_mut() {
                let (op, pcols) = t.inner.to_op(x.len(), y);
                max_pcols = max_pcols.max(pcols);
                self.inner.ops_scratch.push(op);
            }
            self.inner.dispatch_ops(x, max_pcols, 1)
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (x, jobs);
            Err(Error::Unsupported("Metal backend unavailable".into()))
        }
    }

    #[cfg(vxt_has_metal)]
    pub(crate) fn raw_ctx(&self) -> core::ptr::NonNull<core::ffi::c_void> {
        self.inner.ctx
    }

    #[cfg(vxt_has_metal)]
    pub(crate) fn upload_f32_buf(&self, slice: &[f32]) -> Result<crate::metal_ffi::GpuBuf> {
        crate::metal_ffi::GpuBuf::upload(self.inner.ctx, slice)
    }
}

pub struct GpuRaw16Matrix {
    #[cfg(vxt_has_metal)]
    inner: imp::GpuRaw16Inner,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

impl GpuRaw16Matrix {
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

pub struct GpuPackedTensor {
    #[cfg(vxt_has_metal)]
    inner: imp::GpuPackedInner,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

impl GpuPackedTensor {
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

    #[cfg(vxt_has_metal)]
    pub(crate) fn descriptor_op(&self) -> crate::metal_ffi::VxtMetalOp {
        self.inner.raw_op(core::ptr::null_mut()).0
    }
}
