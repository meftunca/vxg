//! Single-Command-Buffer Apple Metal GPU Vision Runner (`MetalVisionRunner`) for
//! `Qwen3_5VisionModel` (`ViT` + 2D Axial `RoPE` + `2x2` `PatchMerger`).

use crate::format::PackedTensor;
use crate::metal::MetalEngine;
use crate::vision_img::VisionConfig;
#[cfg(not(vxt_has_metal))]
use crate::Error;
use crate::Result;

pub(crate) struct VisionBlockBorrow<'a> {
    pub norm1_w: &'a [f32],
    pub norm1_b: &'a [f32],
    pub qkv_w: &'a PackedTensor,
    pub qkv_b: &'a [f32],
    pub proj_w: &'a PackedTensor,
    pub proj_b: &'a [f32],
    pub norm2_w: &'a [f32],
    pub norm2_b: &'a [f32],
    pub fc1_w: &'a PackedTensor,
    pub fc1_b: &'a [f32],
    pub fc2_w: &'a PackedTensor,
    pub fc2_b: &'a [f32],
}

#[cfg(vxt_has_metal)]
mod imp {
    use core::ffi::{c_char, c_void};
    use core::ptr::NonNull;
    use std::sync::Mutex;

    use super::*;
    use crate::fp16::f32_to_f16;
    use crate::metal_ffi::cbuf_to_str;
    use crate::Error;

    #[repr(C)]
    struct VxtVisionBlockUpload {
        norm1_w: *const f32,
        norm1_b: *const f32,
        qkv_w_f16: *const u16,
        qkv_b: *const f32,
        proj_w_f16: *const u16,
        proj_b: *const f32,
        norm2_w: *const f32,
        norm2_b: *const f32,
        fc1_w_f16: *const u16,
        fc1_b: *const f32,
        fc2_w_f16: *const u16,
        fc2_b: *const f32,
    }

    #[repr(C)]
    struct VxtVisionModelUpload {
        depth: u32,
        hidden_size: u32,
        intermediate_size: u32,
        num_heads: u32,
        head_dim: u32,
        patch_dim: u32,
        merge_unit: u32,
        out_hidden_size: u32,
        patch_w_f16: *const u16,
        patch_b: *const f32,
        blocks: *const VxtVisionBlockUpload,
        merger_norm_w: *const f32,
        merger_norm_b: *const f32,
        merger_fc1_w_f16: *const u16,
        merger_fc1_b: *const f32,
        merger_fc2_w_f16: *const u16,
        merger_fc2_b: *const f32,
    }

    extern "C" {
        fn vxt_metal_vision_create(
            gemv_ctx: *mut c_void,
            up: *const VxtVisionModelUpload,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> *mut c_void;
        fn vxt_metal_vision_destroy(runner: *mut c_void);
        fn vxt_metal_vision_encode(
            runner: *mut c_void,
            seq: u32,
            patches_in: *const f32,
            pos_add_in: *const f32,
            rope_cos_in: *const f32,
            rope_sin_in: *const f32,
            embeds_out: *mut f32,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> i32;
    }

    struct RawHandle(NonNull<c_void>);
    unsafe impl Send for RawHandle {}

    impl Drop for RawHandle {
        fn drop(&mut self) {
            unsafe { vxt_metal_vision_destroy(self.0.as_ptr()) };
        }
    }

    fn packed_to_f16_par(t: &PackedTensor, threads: usize) -> Vec<u16> {
        let (rows, cols) = (t.rows(), t.cols());
        let mut out = vec![0u16; rows * cols];
        let workers = threads.clamp(1, 16).min(rows.max(1));
        if workers <= 1 || rows < 32 {
            let mut row_f32 = vec![0.0f32; cols];
            for r in 0..rows {
                t.decode_row(r, &mut row_f32);
                for (dst, &v) in out[r * cols..(r + 1) * cols].iter_mut().zip(&row_f32) {
                    *dst = f32_to_f16(v);
                }
            }
            return out;
        }
        let chunk_r = rows.div_ceil(workers);
        std::thread::scope(|s| {
            for (ci, chunk) in out.chunks_mut(chunk_r * cols).enumerate() {
                let r0 = ci * chunk_r;
                let nr = chunk.len() / cols;
                s.spawn(move || {
                    let mut row_f32 = vec![0.0f32; cols];
                    for dr in 0..nr {
                        t.decode_row(r0 + dr, &mut row_f32);
                        for (dst, &v) in chunk[dr * cols..(dr + 1) * cols].iter_mut().zip(&row_f32)
                        {
                            *dst = f32_to_f16(v);
                        }
                    }
                });
            }
        });
        out
    }

    pub struct RunnerInner {
        handle: Mutex<RawHandle>,
    }

    impl RunnerInner {
        #[allow(clippy::too_many_arguments)]
        pub fn try_create(
            eng: &MetalEngine,
            cfg: &VisionConfig,
            patch_w: &[f32],
            patch_b: &[f32],
            blocks: &[VisionBlockBorrow<'_>],
            merger_norm_w: &[f32],
            merger_norm_b: &[f32],
            merger_fc1_w: &PackedTensor,
            merger_fc1_b: &[f32],
            merger_fc2_w: &PackedTensor,
            merger_fc2_b: &[f32],
            threads: usize,
        ) -> Result<Option<Self>> {
            if cfg.head_dim > 256 || cfg.head_dim % 2 != 0 || blocks.len() != cfg.depth {
                return Ok(None);
            }
            let patch_w_f16: Vec<u16> = patch_w.iter().map(|&x| f32_to_f16(x)).collect();
            let mut blk_f16: Vec<[Vec<u16>; 4]> = Vec::with_capacity(blocks.len());
            for b in blocks {
                blk_f16.push([
                    packed_to_f16_par(b.qkv_w, threads),
                    packed_to_f16_par(b.proj_w, threads),
                    packed_to_f16_par(b.fc1_w, threads),
                    packed_to_f16_par(b.fc2_w, threads),
                ]);
            }
            let m_fc1_f16 = packed_to_f16_par(merger_fc1_w, threads);
            let m_fc2_f16 = packed_to_f16_par(merger_fc2_w, threads);

            let c_blocks: Vec<VxtVisionBlockUpload> = blocks
                .iter()
                .zip(blk_f16.iter())
                .map(|(b, w)| VxtVisionBlockUpload {
                    norm1_w: b.norm1_w.as_ptr(),
                    norm1_b: b.norm1_b.as_ptr(),
                    qkv_w_f16: w[0].as_ptr(),
                    qkv_b: b.qkv_b.as_ptr(),
                    proj_w_f16: w[1].as_ptr(),
                    proj_b: b.proj_b.as_ptr(),
                    norm2_w: b.norm2_w.as_ptr(),
                    norm2_b: b.norm2_b.as_ptr(),
                    fc1_w_f16: w[2].as_ptr(),
                    fc1_b: b.fc1_b.as_ptr(),
                    fc2_w_f16: w[3].as_ptr(),
                    fc2_b: b.fc2_b.as_ptr(),
                })
                .collect();

            let up = VxtVisionModelUpload {
                depth: cfg.depth as u32,
                hidden_size: cfg.hidden_size as u32,
                intermediate_size: cfg.intermediate_size as u32,
                num_heads: cfg.num_heads as u32,
                head_dim: cfg.head_dim as u32,
                patch_dim: cfg.patch_dim() as u32,
                merge_unit: cfg.merge_unit() as u32,
                out_hidden_size: cfg.out_hidden_size as u32,
                patch_w_f16: patch_w_f16.as_ptr(),
                patch_b: patch_b.as_ptr(),
                blocks: c_blocks.as_ptr(),
                merger_norm_w: merger_norm_w.as_ptr(),
                merger_norm_b: merger_norm_b.as_ptr(),
                merger_fc1_w_f16: m_fc1_f16.as_ptr(),
                merger_fc1_b: merger_fc1_b.as_ptr(),
                merger_fc2_w_f16: m_fc2_f16.as_ptr(),
                merger_fc2_b: merger_fc2_b.as_ptr(),
            };

            let mut err = [0u8; 512];
            let raw = unsafe {
                vxt_metal_vision_create(
                    eng.raw_ctx().as_ptr(),
                    &up,
                    err.as_mut_ptr() as *mut c_char,
                    err.len(),
                )
            };
            drop((c_blocks, blk_f16, patch_w_f16, m_fc1_f16, m_fc2_f16));
            let Some(nn) = NonNull::new(raw) else {
                return Err(Error::Backend(cbuf_to_str(&err)));
            };
            Ok(Some(Self {
                handle: Mutex::new(RawHandle(nn)),
            }))
        }

        pub fn encode(
            &self,
            seq: usize,
            patches: &[f32],
            pos_add: &[f32],
            rope_cos: &[f32],
            rope_sin: &[f32],
            embeds_out: &mut [f32],
        ) -> Result<()> {
            let guard = self.handle.lock().unwrap();
            let mut err = [0u8; 512];
            let rc = unsafe {
                vxt_metal_vision_encode(
                    guard.0.as_ptr(),
                    seq as u32,
                    patches.as_ptr(),
                    pos_add.as_ptr(),
                    rope_cos.as_ptr(),
                    rope_sin.as_ptr(),
                    embeds_out.as_mut_ptr(),
                    err.as_mut_ptr() as *mut c_char,
                    err.len(),
                )
            };
            if rc != 0 {
                return Err(Error::Backend(cbuf_to_str(&err)));
            }
            Ok(())
        }
    }
}

pub(crate) struct MetalVisionRunner {
    #[cfg(vxt_has_metal)]
    inner: imp::RunnerInner,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

impl MetalVisionRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn try_create(
        eng: &MetalEngine,
        cfg: &VisionConfig,
        patch_w: &[f32],
        patch_b: &[f32],
        blocks: &[VisionBlockBorrow<'_>],
        merger_norm_w: &[f32],
        merger_norm_b: &[f32],
        merger_fc1_w: &PackedTensor,
        merger_fc1_b: &[f32],
        merger_fc2_w: &PackedTensor,
        merger_fc2_b: &[f32],
        threads: usize,
    ) -> Result<Option<Self>> {
        #[cfg(vxt_has_metal)]
        {
            Ok(imp::RunnerInner::try_create(
                eng,
                cfg,
                patch_w,
                patch_b,
                blocks,
                merger_norm_w,
                merger_norm_b,
                merger_fc1_w,
                merger_fc1_b,
                merger_fc2_w,
                merger_fc2_b,
                threads,
            )?
            .map(|inner| Self { inner }))
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (
                eng,
                cfg,
                patch_w,
                patch_b,
                blocks,
                merger_norm_w,
                merger_norm_b,
                merger_fc1_w,
                merger_fc1_b,
                merger_fc2_w,
                merger_fc2_b,
                threads,
            );
            Ok(None)
        }
    }

    pub fn encode(
        &self,
        seq: usize,
        patches: &[f32],
        pos_add: &[f32],
        rope_cos: &[f32],
        rope_sin: &[f32],
        embeds_out: &mut [f32],
    ) -> Result<()> {
        #[cfg(vxt_has_metal)]
        {
            self.inner
                .encode(seq, patches, pos_add, rope_cos, rope_sin, embeds_out)
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (seq, patches, pos_add, rope_cos, rope_sin, embeds_out);
            Err(Error::Unsupported("Metal vision unavailable".into()))
        }
    }
}
