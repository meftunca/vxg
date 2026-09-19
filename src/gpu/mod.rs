//! Native GPU encode/decode. There is no silent CPU fallback.
//! Pipelines/device buffers persist in `GpuCodec`; frames are batched to bound VRAM.
//! Metal/CUDA/Vulkan SDKs are compile-time requirements of their opt-in features.
use crate::{
    block,
    checksum::crc32c,
    frame::{self, Part},
    Error, Frame, Result,
};
use std::{
    ffi::{c_char, c_void},
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Vulkan,
    Cuda,
}
impl std::str::FromStr for Backend {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "metal" => Ok(Self::Metal),
            "vulkan" | "spirv" => Ok(Self::Vulkan),
            "cuda" => Ok(Self::Cuda),
            _ => Err(Error::InvalidOption(
                "backend must be metal/vulkan/spirv/cuda",
            )),
        }
    }
}
#[derive(Clone, Debug)]
pub struct GpuOptions {
    pub block_size: usize,
    pub checksum: bool,
    /// Bounded device working set. Rounded down to a multiple of the frame's block size.
    pub batch_bytes: usize,
    pub max_output: usize,
    pub verify_checksum: bool,
}
impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            block_size: 65536,
            checksum: true,
            batch_bytes: 16 << 20,
            max_output: 1 << 30,
            verify_checksum: true,
        }
    }
}
#[derive(Clone, Debug)]
pub struct GpuReport {
    pub device: String,
    pub batches: usize,
    pub input_bytes: usize,
    pub output_bytes: usize,
    /// Includes CPU validation/framing/CRC, transfers, allocation/resizing, dispatch and readback.
    /// Excludes `GpuCodec::new` (pipeline creation) and caller's file I/O.
    pub end_to_end_ms: f64,
    /// Sum of device-timestamp dispatch intervals. NOT end-to-end throughput.
    pub kernel_ms: Option<f64>,
}
#[repr(C)]
struct Params {
    total_size: u32,
    block_size: u32,
    block_count: u32,
    operation: u32,
}
type Create = unsafe extern "C" fn(*mut c_char, usize) -> *mut c_void;
type Run = unsafe extern "C" fn(
    *mut c_void,
    *const u8,
    usize,
    *mut u8,
    usize,
    *mut u32,
    usize,
    *const Params,
    *mut f64,
    *mut c_char,
    usize,
) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
#[derive(Clone, Copy)]
struct Api {
    create: Create,
    run: Run,
    destroy: Destroy,
}
#[allow(unused_macros)]
macro_rules! ffi {
    ($module:ident,$create:ident,$run:ident,$destroy:ident) => {
        mod $module {
            use super::*;
            extern "C" {
                pub fn $create(error: *mut c_char, cap: usize) -> *mut c_void;
                pub fn $run(
                    handle: *mut c_void,
                    input: *const u8,
                    input_size: usize,
                    output: *mut u8,
                    output_size: usize,
                    desc: *mut u32,
                    desc_words: usize,
                    params: *const Params,
                    kernel_ms: *mut f64,
                    error: *mut c_char,
                    cap: usize,
                ) -> i32;
                pub fn $destroy(handle: *mut c_void);
            }
        }
    };
}
#[cfg(all(feature = "metal", target_os = "macos"))]
ffi!(metal, vxg_metal_create, vxg_metal_run, vxg_metal_destroy);
#[cfg(feature = "vulkan")]
ffi!(
    vulkan,
    vxg_vulkan_create,
    vxg_vulkan_run,
    vxg_vulkan_destroy
);
#[cfg(feature = "cuda")]
ffi!(cuda, vxg_cuda_create, vxg_cuda_run, vxg_cuda_destroy);
fn api(backend: Backend) -> Result<Api> {
    match backend {
        Backend::Metal => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                return Ok(Api {
                    create: metal::vxg_metal_create,
                    run: metal::vxg_metal_run,
                    destroy: metal::vxg_metal_destroy,
                });
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Err(Error::Unsupported(
                    "build on macOS with --features metal".into(),
                ))
            }
        }
        Backend::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                return Ok(Api {
                    create: vulkan::vxg_vulkan_create,
                    run: vulkan::vxg_vulkan_run,
                    destroy: vulkan::vxg_vulkan_destroy,
                });
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err(Error::Unsupported(
                    "build with --features vulkan and a Vulkan SDK".into(),
                ))
            }
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                return Ok(Api {
                    create: cuda::vxg_cuda_create,
                    run: cuda::vxg_cuda_run,
                    destroy: cuda::vxg_cuda_destroy,
                });
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(Error::Unsupported(
                    "build with --features cuda and a CUDA toolkit".into(),
                ))
            }
        }
    }
}
fn message(buffer: &[c_char]) -> String {
    String::from_utf8_lossy(
        &buffer
            .iter()
            .take_while(|&&x| x != 0)
            .map(|&x| x as u8)
            .collect::<Vec<_>>(),
    )
    .into_owned()
}
pub struct GpuCodec {
    api: Api,
    handle: NonNull<c_void>,
    device: String,
    poisoned: bool,
    // Explicitly !Send + !Sync. The native context is used serially by one caller.
    _not_send: PhantomData<Rc<()>>,
}
impl GpuCodec {
    pub fn new(backend: Backend) -> Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(Error::Unsupported(
                "GPU bridge currently requires a little-endian host".into(),
            ));
        }
        let api = api(backend)?;
        let mut text = [0 as c_char; 2048];
        // SAFETY: native constructor owns the returned context; error buffer is writable.
        let raw = unsafe { (api.create)(text.as_mut_ptr(), text.len()) };
        let handle = NonNull::new(raw).ok_or_else(|| Error::Backend(message(&text)))?;
        Ok(Self {
            api,
            handle,
            device: message(&text),
            poisoned: false,
            _not_send: PhantomData,
        })
    }
    pub fn device_name(&self) -> &str {
        &self.device
    }
    fn invoke(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        desc: &mut [u32],
        params: &Params,
    ) -> Result<Option<f64>> {
        if self.poisoned {
            return Err(Error::Backend(
                "context is poisoned after an earlier GPU error; create a new context".into(),
            ));
        }
        let mut text = [0 as c_char; 2048];
        let mut ms = -1.0;
        // SAFETY: all native calls are synchronous before returning, including failure.
        // Sizes were checked by the caller. Decode commands were fully validated. The
        // context, inputs, outputs and descriptors remain alive and non-aliased throughout.
        let code = unsafe {
            (self.api.run)(
                self.handle.as_ptr(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                desc.as_mut_ptr(),
                desc.len(),
                params,
                &mut ms,
                text.as_mut_ptr(),
                text.len(),
            )
        };
        if code != 0 {
            self.poisoned = true;
            return Err(Error::Backend(message(&text)));
        }
        Ok(if ms.is_finite() && ms >= 0.0 {
            Some(ms)
        } else {
            None
        })
    }
    pub fn compress(&mut self, input: &[u8], options: &GpuOptions) -> Result<(Vec<u8>, GpuReport)> {
        let started = Instant::now();
        frame::check_block_size(options.block_size)?;
        frame::max_compressed_size(input.len(), options.block_size)?;
        let batch = check_batch(options.batch_bytes, options.block_size)?;
        let mut parts = Vec::new();
        let mut batches = 0;
        let mut total_ms = Some(0.0);
        for chunk in input.chunks(batch) {
            let count = chunk.len().div_ceil(options.block_size);
            let mut slots = vec![0u8; count * options.block_size];
            let mut desc = vec![0u32; count * 4];
            let params = Params {
                total_size: chunk.len() as u32,
                block_size: options.block_size as u32,
                block_count: count as u32,
                operation: 0,
            };
            let elapsed = self.invoke(chunk, &mut slots, &mut desc, &params)?;
            total_ms = total_ms.zip(elapsed).map(|(a, b)| a + b);
            batches += 1;
            for (i, raw) in chunk.chunks(options.block_size).enumerate() {
                let size = desc[i * 4] as usize;
                let kind = desc[i * 4 + 1];
                if desc[i * 4 + 2] != 0
                    || desc[i * 4 + 3] != 0
                    || size > raw.len()
                    || kind > 1
                    || (kind == 0 && size != raw.len())
                    || (kind == 1 && size >= raw.len())
                {
                    return Err(Error::Backend(
                        "GPU returned invalid size/kind metadata".into(),
                    ));
                }
                let bytes = slots[i * options.block_size..i * options.block_size + size].to_vec();
                // GPU encoders assemble into the current frame version, which permits
                // opcode 7's compact 16-bit absolute distance.
                block::validate(&bytes, kind, raw.len(), true)?;
                parts.push(Part {
                    bytes,
                    kind,
                    raw_size: raw.len(),
                    checksum: if options.checksum { crc32c(raw) } else { 0 },
                });
            }
        }
        let out = frame::assemble(parts, input.len(), options.block_size, options.checksum)?;
        let report = GpuReport {
            device: self.device.clone(),
            batches,
            input_bytes: input.len(),
            output_bytes: out.len(),
            end_to_end_ms: started.elapsed().as_secs_f64() * 1000.0,
            kernel_ms: total_ms,
        };
        Ok((out, report))
    }
    pub fn decompress(
        &mut self,
        input: &[u8],
        options: &GpuOptions,
    ) -> Result<(Vec<u8>, GpuReport)> {
        let started = Instant::now();
        let frame = Frame::parse(input, options.max_output)?;
        frame.validate_commands()?; // Mandatory before device code: do not remove for a benchmark.
        let batch = check_batch(options.batch_bytes, frame.block_size())?;
        let per_batch = batch / frame.block_size();
        let mut out = Vec::new();
        out.try_reserve_exact(frame.raw_size())
            .map_err(|_| Error::SizeOverflow)?;
        out.resize(frame.raw_size(), 0);
        let mut batches = 0;
        let mut total_ms = Some(0.0);
        let mut first_index = 0;
        for blocks in frame.blocks.chunks(per_batch) {
            let n: usize = blocks.iter().map(|b| b.info.raw_size).sum();
            let mut packed = Vec::new();
            let mut desc = Vec::with_capacity(blocks.len() * 4);
            for b in blocks {
                desc.extend_from_slice(&[
                    packed.len() as u32,
                    b.info.payload_size as u32,
                    b.info.raw_size as u32,
                    b.info.kind,
                ]);
                packed.extend_from_slice(b.payload);
            }
            let mut decoded = vec![0u8; n.div_ceil(4) * 4];
            let params = Params {
                total_size: n as u32,
                block_size: frame.block_size() as u32,
                block_count: blocks.len() as u32,
                operation: 1,
            };
            let elapsed = self.invoke(&packed, &mut decoded, &mut desc, &params)?;
            total_ms = total_ms.zip(elapsed).map(|(a, b)| a + b);
            batches += 1;
            let mut p = 0;
            for (i, b) in blocks.iter().enumerate() {
                let bytes = &decoded[p..p + b.info.raw_size];
                if frame.checksummed()
                    && options.verify_checksum
                    && crc32c(bytes) != b.info.checksum
                {
                    return Err(Error::Checksum {
                        block: first_index + i,
                    });
                }
                out[b.info.raw_offset..b.info.raw_offset + b.info.raw_size].copy_from_slice(bytes);
                p += b.info.raw_size;
            }
            first_index += blocks.len();
        }
        let report = GpuReport {
            device: self.device.clone(),
            batches,
            input_bytes: input.len(),
            output_bytes: out.len(),
            end_to_end_ms: started.elapsed().as_secs_f64() * 1000.0,
            kernel_ms: total_ms,
        };
        Ok((out, report))
    }
}
fn check_batch(bytes: usize, block_size: usize) -> Result<usize> {
    if bytes < block_size || bytes > (256 << 20) {
        return Err(Error::InvalidOption(
            "GPU batch must be >= block size and <=256 MiB",
        ));
    }
    Ok(bytes / block_size * block_size)
}
impl Drop for GpuCodec {
    fn drop(&mut self) {
        // SAFETY: unique owning context, no concurrent or outstanding operation.
        unsafe { (self.api.destroy)(self.handle.as_ptr()) };
    }
}
