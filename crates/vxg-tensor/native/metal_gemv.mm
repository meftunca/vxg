#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "vxt_metal_source.h"

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

struct MetalGemvCtx {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso_raw16 = nil;
    id<MTLComputePipelineState> pso_lossless_bf16 = nil;
    id<MTLComputePipelineState> pso_lossless_fp16 = nil;
    id<MTLComputePipelineState> pso_cb4 = nil;
    id<MTLComputePipelineState> pso_cb3 = nil;
    id<MTLComputePipelineState> pso_q4_0 = nil;

    id<MTLBuffer> x_buf = nil;
    size_t x_cap = 0;
    id<MTLBuffer> y_buf = nil;
    size_t y_cap = 0;

    bool ensure_io(size_t x_bytes, size_t y_bytes) {
        size_t xb = std::max<size_t>(x_bytes, 64);
        size_t yb = std::max<size_t>(y_bytes, 64);
        if (!x_buf || x_cap < xb) {
            x_buf = [device newBufferWithLength:xb options:MTLResourceStorageModeShared];
            if (!x_buf) return false;
            x_cap = xb;
        }
        if (!y_buf || y_cap < yb) {
            y_buf = [device newBufferWithLength:yb options:MTLResourceStorageModeShared];
            if (!y_buf) return false;
            y_cap = yb;
        }
        return true;
    }
};

void write_err(char* err_out, size_t err_cap, const char* msg) {
    if (!err_out || err_cap == 0) return;
    std::snprintf(err_out, err_cap, "%s", msg ? msg : "unknown Metal error");
}

id<MTLComputePipelineState> build_pso(
    id<MTLDevice> dev,
    id<MTLLibrary> lib,
    NSString* fn_name,
    char* err_out,
    size_t err_cap
) {
    id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
    if (!fn) {
        write_err(err_out, err_cap, [[NSString stringWithFormat:@"missing Metal kernel %@", fn_name] UTF8String]);
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        write_err(err_out, err_cap, err ? [[err localizedDescription] UTF8String] : "PSO build failed");
        return nil;
    }
    return pso;
}

int finish_and_read_y(
    id<MTLCommandBuffer> cmd,
    id<MTLBuffer> y_buf,
    float* y_out,
    uint32_t rows,
    double* out_gpu_ms,
    char* err_out,
    size_t err_cap
) {
    CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
    [cmd commit];
    [cmd waitUntilCompleted];
    CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();

    if ([cmd status] != MTLCommandBufferStatusCompleted) {
        NSError* e = [cmd error];
        write_err(err_out, err_cap, e ? [[e localizedDescription] UTF8String] : "MTLCommandBuffer failed");
        return -1;
    }
    double gpu_s = double([cmd GPUEndTime] - [cmd GPUStartTime]);
    if (out_gpu_ms) {
        *out_gpu_ms = (gpu_s > 0.0 ? gpu_s : double(t1 - t0)) * 1000.0;
    }
    std::memcpy(y_out, [y_buf contents], size_t(rows) * sizeof(float));
    return 0;
}

} // namespace

extern "C" {

void* vxt_metal_create(char* dev_name_out, size_t dev_name_cap, char* err_out, size_t err_cap) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            write_err(err_out, err_cap, "no Metal GPU device available");
            return nullptr;
        }
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) {
            write_err(err_out, err_cap, "failed to create MTLCommandQueue");
            return nullptr;
        }

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = NO;

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:VXT_METAL_SOURCE];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            write_err(err_out, err_cap, err ? [[err localizedDescription] UTF8String] : "MSL compile failed");
            return nullptr;
        }

        auto* ctx = new MetalGemvCtx();
        ctx->device = dev;
        ctx->queue = queue;
        ctx->pso_raw16 = build_pso(dev, lib, @"vxt_gemv_raw16", err_out, err_cap);
        ctx->pso_lossless_bf16 = build_pso(dev, lib, @"vxt_gemv_lossless_bf16", err_out, err_cap);
        ctx->pso_lossless_fp16 = build_pso(dev, lib, @"vxt_gemv_lossless_fp16", err_out, err_cap);
        ctx->pso_cb4 = build_pso(dev, lib, @"vxt_gemv_cb4", err_out, err_cap);
        ctx->pso_cb3 = build_pso(dev, lib, @"vxt_gemv_cb3", err_out, err_cap);
        ctx->pso_q4_0 = build_pso(dev, lib, @"vxt_gemv_q4_0", err_out, err_cap);

        if (!ctx->pso_raw16 || !ctx->pso_lossless_bf16 || !ctx->pso_lossless_fp16
            || !ctx->pso_cb4 || !ctx->pso_cb3 || !ctx->pso_q4_0) {
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

int vxt_metal_run_raw16(
    void* raw_ctx,
    void* w_buf_h,
    const float* x_ptr,
    uint32_t cols,
    float* y_ptr,
    uint32_t rows,
    uint32_t is_fp16,
    uint32_t iters,
    double* out_gpu_ms,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<MetalGemvCtx*>(raw_ctx);
        if (!ctx || !w_buf_h || rows == 0 || cols == 0) {
            write_err(err_out, err_cap, "invalid arguments to vxt_metal_run_raw16");
            return -1;
        }
        size_t xb = size_t(cols) * sizeof(float);
        size_t yb = size_t(rows) * sizeof(float);
        if (!ctx->ensure_io(xb, yb)) {
            write_err(err_out, err_cap, "failed to allocate Metal x/y buffers");
            return -1;
        }
        std::memcpy([ctx->x_buf contents], x_ptr, xb);

        id<MTLBuffer> w_buf = (__bridge id<MTLBuffer>)w_buf_h;
        Raw16Params p{rows, cols, is_fp16, 0u};
        MTLSize tg = MTLSizeMake(64, 1, 1);
        MTLSize grid = MTLSizeMake((rows + 1u) / 2u, 1, 1);
        uint32_t runs = std::max<uint32_t>(iters, 1u);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->pso_raw16];
        [enc setBuffer:w_buf offset:0 atIndex:0];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:1];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        for (uint32_t i = 0; i < runs; ++i) {
            if (i > 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
        [enc endEncoding];
        return finish_and_read_y(cmd, ctx->y_buf, y_ptr, rows, out_gpu_ms, err_out, err_cap);
    }
}

int vxt_metal_run_lossless(
    void* raw_ctx,
    void* base_h,
    void* cw_h,
    void* sb_h,
    void* esc_h,
    const float* x_pad_ptr,
    uint32_t padded_cols,
    float* y_ptr,
    uint32_t rows,
    uint32_t is_fp16,
    const uint32_t* lut8,
    uint32_t iters,
    double* out_gpu_ms,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<MetalGemvCtx*>(raw_ctx);
        if (!ctx || !base_h || !cw_h || !sb_h || !esc_h) {
            write_err(err_out, err_cap, "null buffer handle in vxt_metal_run_lossless");
            return -1;
        }
        size_t xb = size_t(padded_cols) * sizeof(float);
        size_t yb = size_t(rows) * sizeof(float);
        if (!ctx->ensure_io(xb, yb)) {
            write_err(err_out, err_cap, "failed to allocate Metal x/y buffers");
            return -1;
        }
        std::memcpy([ctx->x_buf contents], x_pad_ptr, xb);

        LosslessParams p{};
        p.rows = rows;
        p.padded_cols = padded_cols;
        p.is_fp16 = is_fp16;
        for (int i = 0; i < 8; ++i) p.lut[i] = lut8[i];

        id<MTLComputePipelineState> pso = is_fp16 ? ctx->pso_lossless_fp16 : ctx->pso_lossless_bf16;
        MTLSize tg = MTLSizeMake(64, 1, 1);
        MTLSize grid = MTLSizeMake((rows + 1u) / 2u, 1, 1);
        uint32_t runs = std::max<uint32_t>(iters, 1u);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)base_h offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)cw_h offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)sb_h offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)esc_h offset:0 atIndex:3];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:4];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        for (uint32_t i = 0; i < runs; ++i) {
            if (i > 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
        [enc endEncoding];
        return finish_and_read_y(cmd, ctx->y_buf, y_ptr, rows, out_gpu_ms, err_out, err_cap);
    }
}

int vxt_metal_run_codebook(
    void* raw_ctx,
    void* sc_h,
    void* pw_h,
    void* row_ptr_h,
    void* out_h,
    const float* x_pad_ptr,
    uint32_t padded_cols,
    float* y_ptr,
    uint32_t rows,
    uint32_t bits,
    const float* centroids16,
    uint32_t iters,
    double* out_gpu_ms,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<MetalGemvCtx*>(raw_ctx);
        if (!ctx || !sc_h || !pw_h || !row_ptr_h || !out_h) {
            write_err(err_out, err_cap, "null buffer handle in vxt_metal_run_codebook");
            return -1;
        }
        size_t xb = size_t(padded_cols) * sizeof(float);
        size_t yb = size_t(rows) * sizeof(float);
        if (!ctx->ensure_io(xb, yb)) {
            write_err(err_out, err_cap, "failed to allocate Metal x/y buffers");
            return -1;
        }
        std::memcpy([ctx->x_buf contents], x_pad_ptr, xb);

        CodebookParams p{};
        p.rows = rows;
        p.padded_cols = padded_cols;
        p.bits = bits;
        for (int i = 0; i < 16; ++i) p.centroids[i] = centroids16[i];

        id<MTLComputePipelineState> pso = (bits == 4u) ? ctx->pso_cb4 : ctx->pso_cb3;
        MTLSize tg = MTLSizeMake(64, 1, 1);
        MTLSize grid = MTLSizeMake((rows + 1u) / 2u, 1, 1);
        uint32_t runs = std::max<uint32_t>(iters, 1u);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:(__bridge id<MTLBuffer>)sc_h offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)pw_h offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)row_ptr_h offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_h offset:0 atIndex:3];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:4];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        for (uint32_t i = 0; i < runs; ++i) {
            if (i > 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
        [enc endEncoding];
        return finish_and_read_y(cmd, ctx->y_buf, y_ptr, rows, out_gpu_ms, err_out, err_cap);
    }
}

int vxt_metal_run_q4_0(
    void* raw_ctx,
    void* sc_h,
    void* qs_h,
    const float* x_pad_ptr,
    uint32_t padded_cols,
    float* y_ptr,
    uint32_t rows,
    uint32_t iters,
    double* out_gpu_ms,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<MetalGemvCtx*>(raw_ctx);
        if (!ctx || !sc_h || !qs_h) {
            write_err(err_out, err_cap, "null buffer handle in vxt_metal_run_q4_0");
            return -1;
        }
        size_t xb = size_t(padded_cols) * sizeof(float);
        size_t yb = size_t(rows) * sizeof(float);
        if (!ctx->ensure_io(xb, yb)) {
            write_err(err_out, err_cap, "failed to allocate Metal x/y buffers");
            return -1;
        }
        std::memcpy([ctx->x_buf contents], x_pad_ptr, xb);

        Q4_0Params p{rows, padded_cols};
        MTLSize tg = MTLSizeMake(64, 1, 1);
        MTLSize grid = MTLSizeMake((rows + 1u) / 2u, 1, 1);
        uint32_t runs = std::max<uint32_t>(iters, 1u);

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->pso_q4_0];
        [enc setBuffer:(__bridge id<MTLBuffer>)sc_h offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)qs_h offset:0 atIndex:1];
        [enc setBuffer:ctx->x_buf offset:0 atIndex:2];
        [enc setBuffer:ctx->y_buf offset:0 atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        for (uint32_t i = 0; i < runs; ++i) {
            if (i > 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
        [enc endEncoding];
        return finish_and_read_y(cmd, ctx->y_buf, y_ptr, rows, out_gpu_ms, err_out, err_cap);
    }
}

} // extern "C"
