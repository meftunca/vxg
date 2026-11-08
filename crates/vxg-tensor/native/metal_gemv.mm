#include "metal_internal.h"
#include "vxt_metal_source.h"
#include <cstdlib>

namespace {

struct Raw16Params {
    uint32_t rows;
    uint32_t cols;
    uint32_t is_fp16;
    uint32_t _pad;
};

struct LosslessParams {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t is_fp16;
    uint32_t _pad;
    uint32_t lut[8];
};

struct CodebookParams {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t bits;
    uint32_t _pad;
    float centroids[16];
};

struct Q4_0Params {
    uint32_t rows;
    uint32_t padded_cols;
};

struct TurboParams {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t cols;
    uint32_t _pad;
};

struct TurboRotParams {
    uint32_t cols;
    uint32_t padded_cols;
};

} // namespace

void vxt_write_err(char* err_out, size_t err_cap, const char* msg) {
    if (!err_out || err_cap == 0) return;
    std::snprintf(err_out, err_cap, "%s", msg ? msg : "unknown Metal error");
}

id<MTLComputePipelineState> vxt_build_pso(
    id<MTLDevice> dev,
    id<MTLLibrary> lib,
    NSString* fn_name,
    char* err_out,
    size_t err_cap
) {
    id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
    if (!fn) {
        vxt_write_err(
            err_out,
            err_cap,
            [[NSString stringWithFormat:@"missing Metal kernel %@", fn_name] UTF8String]
        );
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        vxt_write_err(
            err_out,
            err_cap,
            err ? [[err localizedDescription] UTF8String] : "PSO build failed"
        );
        return nil;
    }
    return pso;
}

void vxt_encode_fwht32_buf(
    MetalGemvCtx* ctx,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> x_in,
    id<MTLBuffer> x_rot_out,
    uint32_t padded_cols
) {
    uint32_t n_blks = padded_cols >> 5u;
    MTLSize tg = MTLSizeMake(64, 1, 1);
    MTLSize grid = MTLSizeMake((n_blks + 1u) / 2u, 1, 1);
    [enc setComputePipelineState:ctx->pso_fwht32];
    [enc setBuffer:x_in offset:0 atIndex:0];
    [enc setBuffer:x_rot_out offset:0 atIndex:1];
    [enc setBytes:&padded_cols length:sizeof(padded_cols) atIndex:2];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void vxt_encode_turbo_fwht32_buf(
    MetalGemvCtx* ctx,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> x_in,
    id<MTLBuffer> x_rot_out,
    uint32_t cols,
    uint32_t padded_cols
) {
    uint32_t n_blks = padded_cols >> 5u;
    TurboRotParams p{cols > 0u ? cols : padded_cols, padded_cols};
    MTLSize tg = MTLSizeMake(256, 1, 1);
    MTLSize grid = MTLSizeMake((n_blks + 7u) / 8u, 1, 1);
    [enc setComputePipelineState:ctx->pso_turbo_fwht32];
    [enc setBuffer:x_in offset:0 atIndex:0];
    [enc setBuffer:x_rot_out offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
}

void vxt_encode_gemv_op(
    MetalGemvCtx* ctx,
    id<MTLComputeCommandEncoder> enc,
    const VxtMetalOp& op,
    id<MTLBuffer> x_in,
    id<MTLBuffer> x_rot_in,
    id<MTLBuffer> y_out,
    size_t y_byte_off
) {
    MTLSize tg = MTLSizeMake(64, 1, 1);
    MTLSize grid = MTLSizeMake((op.rows + 1u) / 2u, 1, 1);

    switch (op.kind) {
        case 0: {
            Raw16Params p{op.rows, op.cols, op.sub, 0u};
            [enc setComputePipelineState:ctx->pso_raw16];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b0 offset:0 atIndex:0];
            [enc setBuffer:x_in offset:0 atIndex:1];
            [enc setBuffer:y_out offset:y_byte_off atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            break;
        }
        case 1: {
            // Experimental launch grouping only; each row retains one 32-lane SIMD group.
            const char* rows_env = std::getenv("VXG_LOSSLESS_ROWS4");
            const uint32_t rows_per_tg = rows_env && std::strcmp(rows_env, "1") == 0 ? 4u : 2u;
            tg = MTLSizeMake(rows_per_tg * 32u, 1, 1);
            grid = MTLSizeMake((op.rows + rows_per_tg - 1u) / rows_per_tg, 1, 1);
            LosslessParams p{};
            p.rows = op.rows;
            p.padded_cols = op.cols;
            p.is_fp16 = op.sub;
            for (int i = 0; i < 8; ++i) p.lut[i] = op.lut8[i];
            [enc setComputePipelineState:(op.sub ? ctx->pso_lossless_fp16 : ctx->pso_lossless_bf16)];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b0 offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b1 offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b2 offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b3 offset:0 atIndex:3];
            [enc setBuffer:x_in offset:0 atIndex:4];
            [enc setBuffer:y_out offset:y_byte_off atIndex:5];
            [enc setBytes:&p length:sizeof(p) atIndex:6];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            break;
        }
        case 2: {
            if (op.sub == 4u && ctx->cb4_rows4) {
                tg = MTLSizeMake(128, 1, 1);
                grid = MTLSizeMake((op.rows + 3u) / 4u, 1, 1);
            }
            CodebookParams p{};
            p.rows = op.rows;
            p.padded_cols = op.cols;
            p.bits = op.sub;
            for (int i = 0; i < 16; ++i) p.centroids[i] = op.poly16[i];
            [enc setComputePipelineState:((op.sub == 4u) ? ctx->pso_cb4 : ctx->pso_cb3)];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b0 offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b1 offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b2 offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b3 offset:0 atIndex:3];
            [enc setBuffer:x_in offset:0 atIndex:4];
            [enc setBuffer:y_out offset:y_byte_off atIndex:5];
            [enc setBytes:&p length:sizeof(p) atIndex:6];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            break;
        }
        case 3: {
            Q4_0Params p{op.rows, op.cols};
            [enc setComputePipelineState:ctx->pso_q4_0];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b0 offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b1 offset:0 atIndex:1];
            [enc setBuffer:x_in offset:0 atIndex:2];
            [enc setBuffer:y_out offset:y_byte_off atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            break;
        }
        case 4: {
            CodebookParams p{};
            p.rows = op.rows;
            p.padded_cols = op.cols;
            p.bits = 2u;
            for (int i = 0; i < 16; ++i) p.centroids[i] = op.poly16[i];
            [enc setComputePipelineState:ctx->pso_wht2];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b0 offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b1 offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b2 offset:0 atIndex:2];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b3 offset:0 atIndex:3];
            [enc setBuffer:x_in offset:0 atIndex:4];
            [enc setBuffer:x_rot_in offset:0 atIndex:5];
            [enc setBuffer:y_out offset:y_byte_off atIndex:6];
            [enc setBytes:&p length:sizeof(p) atIndex:7];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            break;
        }
        case 5: {
            Q4_0Params p{op.rows, op.cols};
            [enc setComputePipelineState:ctx->pso_raw32];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b0 offset:0 atIndex:0];
            [enc setBuffer:x_in offset:0 atIndex:1];
            [enc setBuffer:y_out offset:y_byte_off atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            break;
        }
        case 6: {
            uint32_t raw_cols = op.lut8[0] > 0u ? op.lut8[0] : op.cols;
            bool is_t2 = (op.sub == 2u);
            bool is_t25 = (op.sub == 5u);
            bool use_nr8 = op.rows >= 4096u;
            uint32_t tg_threads = (is_t2 || is_t25) ? 128u : 64u;
            uint32_t sgs_per_tg = tg_threads >> 5u;
            uint32_t base_rpt = sgs_per_tg * (use_nr8 ? 8u : 4u);
            uint32_t k_loop = (op.rows >= 131072u) ? 2u : 1u;
            uint32_t ec_flag = (op.sub == 4u) ? 256u : 0u;
            TurboParams p{op.rows, op.cols, raw_cols, k_loop | ec_flag};
            id<MTLComputePipelineState> pso = is_t2
                ? (use_nr8 ? ctx->pso_turbo2_nr8 : ctx->pso_turbo2_nr4)
                : (is_t25
                    ? (use_nr8 ? ctx->pso_turbo25_nr8 : ctx->pso_turbo25_nr4)
                    : (use_nr8 ? ctx->pso_turbo3_nr8 : ctx->pso_turbo3_nr4));
            uint32_t rows_per_tg = base_rpt * k_loop;
            MTLSize tg_t = MTLSizeMake(tg_threads, 1, 1);
            MTLSize grid_t = MTLSizeMake((op.rows + rows_per_tg - 1u) / rows_per_tg, 1, 1);
            [enc setComputePipelineState:pso];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b0 offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)op.b1 offset:0 atIndex:1];
            [enc setBuffer:x_rot_in offset:0 atIndex:2];
            [enc setBuffer:y_out offset:y_byte_off atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            [enc dispatchThreadgroups:grid_t threadsPerThreadgroup:tg_t];
            break;
        }
        default:
            break;
    }
}

extern "C" {

void* vxt_metal_create(char* dev_name_out, size_t dev_name_cap, char* err_out, size_t err_cap) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            vxt_write_err(err_out, err_cap, "no Metal GPU device available");
            return nullptr;
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) {
            vxt_write_err(err_out, err_cap, "failed to create MTLCommandQueue");
            return nullptr;
        }

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        MTLCompileOptions* fast_opts = [[MTLCompileOptions alloc] init];
        const NSOperatingSystemVersion mac15 = {15, 0, 0};
        const bool is_mac15 = [[NSProcessInfo processInfo] isOperatingSystemAtLeastVersion:mac15];
        const auto msl_ver = static_cast<MTLLanguageVersion>((3u << 16) + (is_mac15 ? 2u : 0u));
        opts.languageVersion = msl_ver;
        fast_opts.languageVersion = msl_ver;
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = NO;
        fast_opts.fastMathEnabled = YES;
        #pragma clang diagnostic pop

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:VXT_METAL_SOURCE];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            vxt_write_err(err_out, err_cap, err ? [[err localizedDescription] UTF8String] : "MSL compile failed");
            return nullptr;
        }
        id<MTLLibrary> fast_lib = [dev newLibraryWithSource:src options:fast_opts error:nil];

        auto* ctx = new MetalGemvCtx();
        ctx->device = dev;
        ctx->queue = queue;
        ctx->lib = lib;
        ctx->fast_lib = fast_lib ? fast_lib : lib;
        ctx->pso_raw16 = vxt_build_pso(dev, lib, @"vxt_gemv_raw16", err_out, err_cap);
        ctx->pso_raw32 = vxt_build_pso(dev, lib, @"vxt_gemv_raw32", err_out, err_cap);
        ctx->pso_lossless_bf16 = vxt_build_pso(dev, lib, @"vxt_gemv_lossless_bf16", err_out, err_cap);
        ctx->pso_lossless_fp16 = vxt_build_pso(dev, lib, @"vxt_gemv_lossless_fp16", err_out, err_cap);
        const char* rows4 = std::getenv("VXG_CB4_ROWS4");
        ctx->cb4_rows4 = rows4 && std::strcmp(rows4, "1") == 0;
        ctx->pso_cb4 = vxt_build_pso(dev, lib, ctx->cb4_rows4 ? @"vxt_gemv_cb4_rows4" : @"vxt_gemv_cb4", err_out, err_cap);
        ctx->pso_cb3 = vxt_build_pso(dev, lib, @"vxt_gemv_cb3", err_out, err_cap);
        ctx->pso_q4_0 = vxt_build_pso(dev, lib, @"vxt_gemv_q4_0", err_out, err_cap);
        ctx->pso_fwht32 = vxt_build_pso(dev, lib, @"vxt_fwht32_vec", err_out, err_cap);
        ctx->pso_wht2 = vxt_build_pso(dev, lib, @"vxt_gemv_wht2", err_out, err_cap);
        ctx->pso_turbo_fwht32 = vxt_build_pso(dev, lib, @"vxt_turbo_fwht32_vec", err_out, err_cap);
        ctx->pso_turbo2_nr4 = vxt_build_pso(dev, lib, @"vxt_gemv_turbo2_nr4", err_out, err_cap);
        ctx->pso_turbo2_nr8 = vxt_build_pso(dev, lib, @"vxt_gemv_turbo2_nr8", err_out, err_cap);
        ctx->pso_turbo25_nr4 = vxt_build_pso(dev, lib, @"vxt_gemv_turbo25_nr4", err_out, err_cap);
        ctx->pso_turbo25_nr8 = vxt_build_pso(dev, lib, @"vxt_gemv_turbo25_nr8", err_out, err_cap);
        ctx->pso_turbo3_nr4 = vxt_build_pso(dev, lib, @"vxt_gemv_turbo3_nr4", err_out, err_cap);
        ctx->pso_turbo3_nr8 = vxt_build_pso(dev, lib, @"vxt_gemv_turbo3_nr8", err_out, err_cap);

        if (!ctx->pso_raw16 || !ctx->pso_raw32 || !ctx->pso_lossless_bf16 || !ctx->pso_lossless_fp16
            || !ctx->pso_cb4 || !ctx->pso_cb3 || !ctx->pso_q4_0
            || !ctx->pso_fwht32 || !ctx->pso_wht2 || !ctx->pso_turbo_fwht32
            || !ctx->pso_turbo2_nr4 || !ctx->pso_turbo2_nr8
            || !ctx->pso_turbo25_nr4 || !ctx->pso_turbo25_nr8
            || !ctx->pso_turbo3_nr4 || !ctx->pso_turbo3_nr8) {
            delete ctx;
            return nullptr;
        }

        if (dev_name_out && dev_name_cap > 0) {
            const char* nm = [[dev name] UTF8String];
            std::snprintf(dev_name_out, dev_name_cap, "%s", nm ? nm : "Apple Metal GPU");
        }
        return ctx;
    }
}

void vxt_metal_destroy(void* raw_ctx) {
    if (!raw_ctx) return;
    @autoreleasepool {
        auto* ctx = static_cast<MetalGemvCtx*>(raw_ctx);
        delete ctx;
    }
}

void* vxt_metal_upload(void* raw_ctx, const void* data, size_t byte_len) {
    if (!raw_ctx) return nullptr;
    @autoreleasepool {
        auto* ctx = static_cast<MetalGemvCtx*>(raw_ctx);
        size_t alloc_len = std::max<size_t>(byte_len, 64);
        id<MTLBuffer> buf = [ctx->device newBufferWithLength:alloc_len
                                                     options:MTLResourceStorageModeShared];
        if (!buf) return nullptr;
        std::memset([buf contents], 0, alloc_len);
        if (data && byte_len > 0) {
            std::memcpy([buf contents], data, byte_len);
        }
        return (__bridge_retained void*)buf;
    }
}

void vxt_metal_free_buffer(void* buf_handle) {
    if (!buf_handle) return;
    @autoreleasepool {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buf_handle;
        (void)buf;
    }
}

int vxt_metal_run_multi(
    void* raw_ctx,
    const float* x_pad_ptr,
    uint32_t x_pad_elems,
    const VxtMetalOp* ops,
    uint32_t n_ops,
    uint32_t iters,
    double* out_gpu_ms,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<MetalGemvCtx*>(raw_ctx);
        if (!ctx || !x_pad_ptr || !ops || n_ops == 0) {
            vxt_write_err(err_out, err_cap, "invalid arguments to vxt_metal_run_multi");
            return -1;
        }

        bool need_rot = false;
        bool need_turbo = false;
        uint32_t max_rot_cols = 0;
        uint32_t max_turbo_cols = 0;
        size_t total_y_aligned = 0;
        for (uint32_t i = 0; i < n_ops; ++i) {
            if (ops[i].kind == 4u) {
                need_rot = true;
                max_rot_cols = std::max(max_rot_cols, ops[i].cols);
            } else if (ops[i].kind == 6u) {
                need_turbo = true;
                max_turbo_cols = std::max(max_turbo_cols, ops[i].cols);
            }
            size_t r_aligned = ((size_t(ops[i].rows) + 63u) & ~size_t(63u));
            total_y_aligned += r_aligned;
        }

        uint32_t max_cols = std::max(x_pad_elems, std::max(max_rot_cols, max_turbo_cols));
        size_t xb = size_t(max_cols) * sizeof(float);
        size_t yb = total_y_aligned * sizeof(float);
        if (!ctx->ensure_io(xb, yb, need_rot, need_turbo)) {
            vxt_write_err(err_out, err_cap, "failed to allocate Metal x/y buffers");
            return -1;
        }
        std::memcpy([ctx->x_buf contents], x_pad_ptr, size_t(x_pad_elems) * sizeof(float));
        if (xb > size_t(x_pad_elems) * sizeof(float)) {
            std::memset(
                static_cast<uint8_t*>([ctx->x_buf contents]) + size_t(x_pad_elems) * sizeof(float),
                0,
                xb - size_t(x_pad_elems) * sizeof(float)
            );
        }

        uint32_t runs = std::max<uint32_t>(iters, 1u);
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        for (uint32_t r = 0; r < runs; ++r) {
            if (need_rot) {
                vxt_encode_fwht32_buf(ctx, enc, ctx->x_buf, ctx->x_rot_buf, max_rot_cols);
            }
            if (need_turbo) {
                vxt_encode_turbo_fwht32_buf(
                    ctx, enc, ctx->x_buf, ctx->x_turbo_buf, x_pad_elems, max_turbo_cols
                );
            }
            size_t y_elem_off = 0;
            for (uint32_t i = 0; i < n_ops; ++i) {
                id<MTLBuffer> rot_b = (ops[i].kind == 6u) ? ctx->x_turbo_buf : ctx->x_rot_buf;
                vxt_encode_gemv_op(
                    ctx, enc, ops[i], ctx->x_buf, rot_b, ctx->y_buf, y_elem_off * sizeof(float)
                );
                y_elem_off += ((size_t(ops[i].rows) + 63u) & ~size_t(63u));
            }
        }
        [enc endEncoding];

        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        [cmd commit];
        while ([cmd status] < MTLCommandBufferStatusCompleted) {
            if (CFAbsoluteTimeGetCurrent() - t0 > 0.004) {
                [cmd waitUntilCompleted];
                break;
            }
#if defined(__aarch64__)
            __builtin_arm_yield();
#endif
        }
        CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            NSError* e = [cmd error];
            vxt_write_err(err_out, err_cap, e ? [[e localizedDescription] UTF8String] : "MTLCommandBuffer failed");
            return -1;
        }
        double gpu_s = double([cmd GPUEndTime] - [cmd GPUStartTime]);
        if (out_gpu_ms) {
            *out_gpu_ms = (gpu_s > 0.0 ? gpu_s : double(t1 - t0)) * 1000.0;
        }

        const float* y_src = static_cast<const float*>([ctx->y_buf contents]);
        size_t y_elem_off = 0;
        for (uint32_t i = 0; i < n_ops; ++i) {
            std::memcpy(ops[i].y_dst, y_src + y_elem_off, size_t(ops[i].rows) * sizeof(float));
            y_elem_off += ((size_t(ops[i].rows) + 63u) & ~size_t(63u));
        }
        return 0;
    }
}

} // extern "C"
