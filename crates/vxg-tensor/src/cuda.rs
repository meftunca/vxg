//! Safe Rust wrapper around the native NVIDIA CUDA zero-LUT `E8` (`bits = 7`),
//! `E8-HQ` (`bits = 8`), and `RawF32` GEMV backend (`VXG_CUDA=1` on Linux/NVIDIA).
//! Compiles to zero-cost stubs on non-CUDA platforms (e.g., macOS / Apple Metal).

use crate::format::PackedTensor;
use crate::{Error, Result};

#[cfg(vxt_has_cuda)]
mod ffi {
    use std::ffi::{c_char, c_int, c_void};

    extern "C" {
        pub fn vxt_cuda_device_count() -> c_int;
        pub fn vxt_cuda_device_name(dev: c_int, buf: *mut c_char, cap: usize) -> c_int;
        pub fn vxt_cuda_ctx_create() -> *mut c_void;
        pub fn vxt_cuda_ctx_destroy(ctx: *mut c_void);
        pub fn vxt_cuda_upload_turbo(
            ctx: *mut c_void,
            bits: u8,
            rows: u32,
            cols: u32,
            padded_cols: u32,
            scales_f16: *const u16,
            n_scales: usize,
            packed_words: *const u32,
            n_words: usize,
        ) -> *mut c_void;
        pub fn vxt_cuda_upload_f32(
            ctx: *mut c_void,
            rows: u32,
            cols: u32,
            data: *const f32,
        ) -> *mut c_void;
        pub fn vxt_cuda_free_tensor(t: *mut c_void);
        pub fn vxt_cuda_gemv(
            ctx: *mut c_void,
            t: *mut c_void,
            x_host: *const f32,
            y_host: *mut f32,
        ) -> c_int;
        pub fn vxt_cuda_gemv_multi(
            ctx: *mut c_void,
            n_jobs: usize,
            tensors: *const *mut c_void,
            x_host: *const f32,
            y_hosts: *const *mut f32,
        ) -> c_int;
    }
}

/// Resident NVIDIA GPU tensor buffer (`E8`, `E8-HQ`, or `RawF32`).
pub struct CudaPackedTensor {
    #[cfg(vxt_has_cuda)]
    raw: *mut std::ffi::c_void,
    rows: usize,
    cols: usize,
}

#[cfg(vxt_has_cuda)]
unsafe impl Send for CudaPackedTensor {}
#[cfg(vxt_has_cuda)]
unsafe impl Sync for CudaPackedTensor {}

impl Drop for CudaPackedTensor {
    fn drop(&mut self) {
        #[cfg(vxt_has_cuda)]
        if !self.raw.is_null() {
            unsafe { ffi::vxt_cuda_free_tensor(self.raw); }
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Native NVIDIA CUDA execution context with reusable stream and scratch buffers.
pub struct CudaEngine {
    #[cfg(vxt_has_cuda)]
    raw: *mut std::ffi::c_void,
    name: String,
}

#[cfg(vxt_has_cuda)]
unsafe impl Send for CudaEngine {}
#[cfg(vxt_has_cuda)]
unsafe impl Sync for CudaEngine {}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        #[cfg(vxt_has_cuda)]
        if !self.raw.is_null() {
            unsafe { ffi::vxt_cuda_ctx_destroy(self.raw); }
            self.raw = std::ptr::null_mut();
        }
    }
}

impl CudaEngine {
    /// Initializes CUDA device 0 if compiled with `VXG_CUDA=1` and a compatible NVIDIA GPU is present.
    pub fn new() -> Result<Self> {
        #[cfg(vxt_has_cuda)]
        {
            let count = unsafe { ffi::vxt_cuda_device_count() };
            if count <= 0 {
                return Err(Error::Unsupported("no NVIDIA CUDA devices detected".into()));
            }
            let raw = unsafe { ffi::vxt_cuda_ctx_create() };
            if raw.is_null() {
                return Err(Error::Unsupported("failed to create CUDA context".into()));
            }
            let mut buf = [0u8; 256];
            let name = if unsafe { ffi::vxt_cuda_device_name(0, buf.as_mut_ptr() as *mut _, buf.len()) } == 0 {
                let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
                String::from_utf8_lossy(&buf[..nul]).trim().to_owned()
            } else {
                "NVIDIA CUDA GPU".to_owned()
            };
            Ok(Self { raw, name })
        }
        #[cfg(not(vxt_has_cuda))]
        {
            Err(Error::Unsupported("CUDA backend not compiled (build on Linux with VXG_CUDA=1)".into()))
        }
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Uploads a `TurboWht(bits = 7 | 8)` or `RawF32` matrix into NVIDIA VRAM.
    /// Returns `Ok(None)` for unsupported legacy packed kinds so callers cleanly fall back to CPU.
    pub fn upload_packed(&self, tensor: &PackedTensor) -> Result<Option<CudaPackedTensor>> {
        #[cfg(vxt_has_cuda)]
        {
            match tensor {
                PackedTensor::TurboWht(m) if matches!(m.bits, 7 | 8) => {
                    let raw = unsafe {
                        ffi::vxt_cuda_upload_turbo(
                            self.raw,
                            m.bits,
                            m.rows as u32,
                            m.cols as u32,
                            m.padded_cols as u32,
                            m.scales_f16.as_ptr(),
                            m.scales_f16.len(),
                            m.packed_words.as_ptr(),
                            m.packed_words.len(),
                        )
                    };
                    if raw.is_null() {
                        return Err(Error::Io("CUDA upload_turbo failed (out of VRAM?)".into()));
                    }
                    Ok(Some(CudaPackedTensor { raw, rows: m.rows, cols: m.cols }))
                }
                PackedTensor::RawF32 { rows, cols, data } if *rows >= 4 && *cols >= 64 => {
                    let raw = unsafe { ffi::vxt_cuda_upload_f32(self.raw, *rows as u32, *cols as u32, data.as_ptr()) };
                    if raw.is_null() {
                        return Err(Error::Io("CUDA upload_f32 failed".into()));
                    }
                    Ok(Some(CudaPackedTensor { raw, rows: *rows, cols: *cols }))
                }
                _ => Ok(None),
            }
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = tensor;
            Ok(None)
        }
    }

    /// Executes a single `y = W * x` compressed-domain GEMV on the NVIDIA GPU.
    pub fn run_gemv(&self, t: &CudaPackedTensor, x: &[f32], y: &mut [f32]) -> Result<()> {
        if x.len() < t.cols || y.len() < t.rows {
            return Err(Error::InvalidShape("CUDA GEMV buffer dimension mismatch"));
        }
        #[cfg(vxt_has_cuda)]
        {
            let rc = unsafe { ffi::vxt_cuda_gemv(self.raw, t.raw, x.as_ptr(), y.as_mut_ptr()) };
            if rc != 0 {
                return Err(Error::Io(format!("vxt_cuda_gemv failed with code {rc}")));
            }
            Ok(())
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (t, x, y);
            Err(Error::Unsupported("CUDA backend not compiled".into()))
        }
    }

    /// Executes multiple projections sharing the same `x` (rotates `x` once via warp-shuffle FWHT-32).
    pub fn run_multi_same_x(&self, x: &[f32], jobs: &mut [(&CudaPackedTensor, &mut [f32])]) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        #[cfg(vxt_has_cuda)]
        {
            let mut ptrs = Vec::with_capacity(jobs.len());
            let mut ys = Vec::with_capacity(jobs.len());
            for (t, y) in jobs.iter_mut() {
                if x.len() < t.cols || y.len() < t.rows {
                    return Err(Error::InvalidShape("CUDA multi-GEMV dimension mismatch"));
                }
                ptrs.push(t.raw);
                ys.push(y.as_mut_ptr());
            }
            let rc = unsafe {
                ffi::vxt_cuda_gemv_multi(self.raw, jobs.len(), ptrs.as_ptr(), x.as_ptr(), ys.as_ptr())
            };
            if rc != 0 {
                return Err(Error::Io(format!("vxt_cuda_gemv_multi failed with code {rc}")));
            }
            Ok(())
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (x, jobs);
            Err(Error::Unsupported("CUDA backend not compiled".into()))
        }
    }
}

/// Uploads all compatible `TurboWht(bits=7|8)` and `RawF32` model matrices to NVIDIA VRAM.
pub(crate) fn upload_model_cuda(
    eng: &CudaEngine,
    token_embd: &mut crate::llm_cpu::MatWeight,
    output_head: &mut Option<crate::llm_cpu::MatWeight>,
    layers: &mut [crate::llm_cpu::DecoderLayer],
) -> Result<()> {
    use crate::llm_cpu::LayerAttn;
    token_embd.upload_cuda(eng)?;
    if let Some(h) = output_head.as_mut() {
        h.upload_cuda(eng)?;
    }
    for ly in layers.iter_mut() {
        match &mut ly.attn {
            LayerAttn::LinearDeltaNet { wqkv, wqkv_gate, ssm_alpha, ssm_beta, ssm_out, .. } => {
                wqkv.upload_cuda(eng)?;
                wqkv_gate.upload_cuda(eng)?;
                ssm_alpha.upload_cuda(eng)?;
                ssm_beta.upload_cuda(eng)?;
                ssm_out.upload_cuda(eng)?;
            }
            LayerAttn::FullGqa { wq, wk, wv, wo, .. } => {
                wq.upload_cuda(eng)?;
                wk.upload_cuda(eng)?;
                wv.upload_cuda(eng)?;
                wo.upload_cuda(eng)?;
            }
        }
        if let Some(moe) = ly.ffn_moe.as_mut() {
            moe.upload_cuda(eng)?;
        } else {
            ly.ffn_gate.upload_cuda(eng)?;
            ly.ffn_up.upload_cuda(eng)?;
            ly.ffn_down.upload_cuda(eng)?;
        }
    }
    Ok(())
}

#[cfg(all(test, vxt_has_cuda))]
mod tests {
    use super::CudaEngine;
    use crate::format::PackedTensor;
    use crate::wht_turbo::TurboWhtMatrix;

    #[test]
    fn cuda_gemv_e8_and_e8_hq_match_cpu() {
        let eng = CudaEngine::new().expect("CUDA device required when VXG_CUDA=1");
        let (rows, cols) = (132usize, 512usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.01).collect();
        let x: Vec<f32> = (0..cols).map(|i| (((i * 17) % 53) as f32 - 26.0) * 0.04).collect();
        for bits in [7u8, 8u8] {
            let m = TurboWhtMatrix::encode_with_ch_amp_threads(bits, rows, cols, &w, None, None, |_| 2.0, 2).unwrap();
            let pt = PackedTensor::TurboWht(m);
            let mut y_cpu = vec![0.0f32; rows];
            let mut y_gpu = vec![0.0f32; rows];
            pt.gemv_cpu(&x, &mut y_cpu, 2);
            let ct = eng.upload_packed(&pt).unwrap().expect("TurboWht bits=7|8 must upload to CUDA");
            eng.run_gemv(&ct, &x, &mut y_gpu).unwrap();
            let max_err = y_cpu.iter().zip(&y_gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_err < 1e-3, "CUDA bits={bits} parity max_err={max_err}");
        }
    }
}
