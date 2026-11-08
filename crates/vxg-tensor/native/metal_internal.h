#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {

struct VxtMetalOp {
    uint32_t kind; // 0=raw16, 1=lossless, 2=codebook, 3=q4_0, 4=wht2, 5=raw32
    uint32_t sub;  // is_fp16 (0,1) or bits (2)
    uint32_t rows;
    uint32_t cols; // cols (raw16/raw32) or padded_cols (packed)
    uint32_t lut8[8];
    float poly16[16];
    void* b0;
    void* b1;
    void* b2;
    void* b3;
    float* y_dst;
};

struct VxtLlmConfig {
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t rope_dim;
    uint32_t rope_neox;
    float rope_freq_base;
    float rms_norm_eps;
    uint32_t ssm_conv_kernel;
    uint32_t ssm_state_size;
    uint32_t ssm_group_count;
    uint32_t ssm_inner_size;
    uint32_t ssm_dt_rank;
};

struct VxtLlmLayerDesc {
    uint32_t is_recurrent;
    uint32_t gated_q;
    void* attn_norm_buf;
    void* post_attn_norm_buf;
    VxtMetalOp op_a0;
    VxtMetalOp op_a1;
    VxtMetalOp op_a2;
    VxtMetalOp op_a3;
    VxtMetalOp op_out;
    void* vec0_buf;
    void* vec1_buf;
    void* vec2_buf;
    void* vec3_buf;
    void* vec4_buf;
    VxtMetalOp op_ffn_gate;
    VxtMetalOp op_ffn_up;
    VxtMetalOp op_ffn_down;
};

struct VxtTurboMat {
    uint32_t bits;
    uint32_t rows;
    uint32_t cols;
    uint32_t padded_cols;
    void* scales_buf;
    void* packed_buf;
};

struct VxtTurboLayerDesc {
    uint32_t is_recurrent;
    uint32_t gated_q;
    uint32_t has_split_gate;
    uint32_t gqa_flags;
    void* attn_norm_buf;
    void* post_attn_norm_buf;
    VxtTurboMat w_qkv;
    VxtTurboMat w_gate_sep;
    void* ab_rot_f16_buf;
    VxtTurboMat w_out;
    void* vec0_buf;
    void* vec1_buf;
    void* vec2_buf;
    void* vec3_buf;
    void* vec4_buf;
    VxtTurboMat w_ffn_gu;
    VxtTurboMat w_ffn_down;
};

} // extern "C"

struct MetalGemvCtx {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;
    id<MTLLibrary> fast_lib = nil;
    id<MTLComputePipelineState> pso_raw16 = nil;
    id<MTLComputePipelineState> pso_raw32 = nil;
    id<MTLComputePipelineState> pso_lossless_bf16 = nil;
    id<MTLComputePipelineState> pso_lossless_fp16 = nil;
    id<MTLComputePipelineState> pso_cb4 = nil;
    bool cb4_rows4 = false;
    id<MTLComputePipelineState> pso_cb3 = nil;
    id<MTLComputePipelineState> pso_q4_0 = nil;
    id<MTLComputePipelineState> pso_fwht32 = nil;
    id<MTLComputePipelineState> pso_wht2 = nil;
    id<MTLComputePipelineState> pso_turbo_fwht32 = nil;
    id<MTLComputePipelineState> pso_turbo2_nr4 = nil;
    id<MTLComputePipelineState> pso_turbo2_nr8 = nil;
    id<MTLComputePipelineState> pso_turbo25_nr4 = nil;
    id<MTLComputePipelineState> pso_turbo25_nr8 = nil;
    id<MTLComputePipelineState> pso_turbo3_nr4 = nil;
    id<MTLComputePipelineState> pso_turbo3_nr8 = nil;

    id<MTLBuffer> x_buf = nil;
    size_t x_cap = 0;
    id<MTLBuffer> x_rot_buf = nil;
    size_t x_rot_cap = 0;
    id<MTLBuffer> x_turbo_buf = nil;
    size_t x_turbo_cap = 0;
    id<MTLBuffer> y_buf = nil;
    size_t y_cap = 0;

    bool ensure_io(size_t x_bytes, size_t y_bytes, bool need_rot, bool need_turbo = false) {
        size_t xb = std::max<size_t>(x_bytes, 64);
        size_t yb = std::max<size_t>(y_bytes, 64);
        if (!x_buf || x_cap < xb) {
            x_buf = [device newBufferWithLength:xb options:MTLResourceStorageModeShared];
            if (!x_buf) return false;
            x_cap = xb;
        }
        if (need_rot && (!x_rot_buf || x_rot_cap < xb)) {
            x_rot_buf = [device newBufferWithLength:xb options:MTLResourceStorageModeShared];
            if (!x_rot_buf) return false;
            x_rot_cap = xb;
        }
        if (need_turbo && (!x_turbo_buf || x_turbo_cap < xb)) {
            x_turbo_buf = [device newBufferWithLength:xb options:MTLResourceStorageModeShared];
            if (!x_turbo_buf) return false;
            x_turbo_cap = xb;
        }
        if (!y_buf || y_cap < yb) {
            y_buf = [device newBufferWithLength:yb options:MTLResourceStorageModeShared];
            if (!y_buf) return false;
            y_cap = yb;
        }
        return true;
    }
};

void vxt_write_err(char* err_out, size_t err_cap, const char* msg);

id<MTLComputePipelineState> vxt_build_pso(
    id<MTLDevice> dev,
    id<MTLLibrary> lib,
    NSString* fn_name,
    char* err_out,
    size_t err_cap
);

void vxt_encode_fwht32_buf(
    MetalGemvCtx* ctx,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> x_in,
    id<MTLBuffer> x_rot_out,
    uint32_t padded_cols
);

void vxt_encode_turbo_fwht32_buf(
    MetalGemvCtx* ctx,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> x_in,
    id<MTLBuffer> x_rot_out,
    uint32_t cols,
    uint32_t padded_cols
);

void vxt_encode_gemv_op(
    MetalGemvCtx* ctx,
    id<MTLComputeCommandEncoder> enc,
    const VxtMetalOp& op,
    id<MTLBuffer> x_in,
    id<MTLBuffer> x_rot_in,
    id<MTLBuffer> y_out,
    size_t y_byte_off
);
