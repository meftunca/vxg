#pragma once

#include "metal_internal.h"
#include <cmath>
#include <cstdlib>
#include <vector>

static constexpr uint32_t CB4_MAX_BATCH_M = 32u;
static constexpr uint32_t CB4_RING_SLOTS = 256u;
static constexpr uint32_t CB4_ARGMAX_TGS = 256u;

static inline size_t cb4_align256(size_t n) {
    return (std::max<size_t>(n, 16) + 255u) & ~size_t(255u);
}

static inline uint32_t cb4_pad64(uint32_t n) {
    return (n + 63u) & ~63u;
}

static inline uint32_t cb4_pad256(uint32_t n) {
    return (n + 255u) & ~255u;
}

static inline id<MTLBuffer> cb4_alloc_zeroed(id<MTLDevice> dev, size_t bytes) {
    size_t sz = std::max<size_t>(bytes, 64);
    id<MTLBuffer> b = [dev newBufferWithLength:sz options:MTLResourceStorageModeShared];
    if (b) std::memset([b contents], 0, sz);
    return b;
}

struct Cb4ArenaMat {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t padded_cols = 0;
    uint32_t split0 = 0;
    uint32_t split1 = 0;
    bool is_nr4 = false;
    float poly[3][4] = {};
    size_t sc_off = 0;
    size_t pw_off = 0;
    size_t rp_off = 0;
    size_t out_off = 0;
};

struct Cb4ArenaLayer {
    uint32_t is_recurrent = 0;
    uint32_t gated_q = 0;
    uint32_t ab_is_strided32 = 0;
    uint32_t gqa_flags = 0;
    size_t attn_norm_off = 0;
    size_t post_attn_norm_off = 0;
    Cb4ArenaMat w_in;
    Cb4ArenaMat w_out;
    Cb4ArenaMat w_ffn_gu;
    Cb4ArenaMat w_ffn_dn;
    size_t ab_off = 0;
    size_t vec_off[5] = {0, 0, 0, 0, 0};
    bool has_vec[5] = {false, false, false, false, false};
};

struct MetalCb4FastRunner {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    VxtLlmConfig cfg{};
    uint32_t dispatch_mode = 0;
    bool disable_fusions = false;
    bool profile_enabled = false;

    id<MTLComputePipelineState> pso_embed = nil;
    id<MTLComputePipelineState> pso_embed_norm0 = nil;
    id<MTLComputePipelineState> pso_rms_norm = nil;
    id<MTLComputePipelineState> pso_add_rms_norm = nil;
    id<MTLComputePipelineState> pso_swiglu = nil;
    id<MTLComputePipelineState> pso_ab_gemv = nil;
    id<MTLComputePipelineState> pso_dnet_prep_m1 = nil;
    id<MTLComputePipelineState> pso_dnet_state_m1 = nil;
    id<MTLComputePipelineState> pso_dnet_out_m1 = nil;
    id<MTLComputePipelineState> pso_dnet_conv_batch = nil;
    id<MTLComputePipelineState> pso_dnet_heads_batch = nil;
    id<MTLComputePipelineState> pso_gqa_fused_m1 = nil;
    id<MTLComputePipelineState> pso_gqa_prep_batch = nil;
    id<MTLComputePipelineState> pso_gqa_attn_batch = nil;
    id<MTLComputePipelineState> pso_gemv_nr4 = nil;
    id<MTLComputePipelineState> pso_gemv_nr2 = nil;
    id<MTLComputePipelineState> pso_ffn_gu_m1 = nil;
    id<MTLComputePipelineState> pso_gemm_nr2 = nil;
    id<MTLComputePipelineState> pso_argmax_s1 = nil;
    id<MTLComputePipelineState> pso_argmax_s2 = nil;

    id<MTLBuffer> weight_arena_buf = nil;
    id<MTLBuffer> dummy_buf = nil;

    Cb4ArenaMat embed_mat;
    float embed_centroids[16] = {};
    Cb4ArenaMat head_mat;
    size_t output_norm_off = 0;
    std::vector<Cb4ArenaLayer> layers;

    uint32_t padded_embd = 0;
    uint32_t padded_ff = 0;
    uint32_t padded_inner = 0;
    uint32_t padded_att = 0;
    uint32_t qkv_z_stride = 0;
    uint32_t conv_stride = 0;
    uint32_t gu_stride = 0;
    uint32_t ab_half_stride = 0;
    uint32_t ab_stride = 0;

    id<MTLBuffer> tok_in_buf = nil;
    id<MTLBuffer> mrope_pos_buf = nil;
    id<MTLBuffer> tok_ring_buf = nil;
    id<MTLBuffer> out_ring_buf = nil;
    id<MTLBuffer> err_ring_buf = nil;
    id<MTLBuffer> s1_vals_buf = nil;
    id<MTLBuffer> s1_ids_buf = nil;
    id<MTLBuffer> s1_bad_buf = nil;

    id<MTLBuffer> cur_buf = nil;
    id<MTLBuffer> delta_buf = nil;
    id<MTLBuffer> normed_buf = nil;
    id<MTLBuffer> qkv_z_buf = nil;
    id<MTLBuffer> ab_buf = nil;
    id<MTLBuffer> conv_silu_buf = nil;
    id<MTLBuffer> o_raw_buf = nil;
    id<MTLBuffer> y_concat_buf = nil;
    id<MTLBuffer> att_cat_buf = nil;
    id<MTLBuffer> q_cur_buf = nil;
    id<MTLBuffer> q_gate_buf = nil;
    id<MTLBuffer> ffn_gu_buf = nil;
    id<MTLBuffer> ffn_act_buf = nil;
    id<MTLBuffer> logits_buf = nil;
};

struct MetalCb4FastState {
    id<MTLDevice> device = nil;
    uint32_t n_layer = 0;
    uint32_t kv_stride = 0;
    uint32_t cap_seq = 0;
    std::vector<uint8_t> is_rec;
    std::vector<size_t> conv_bytes;
    std::vector<size_t> ssm_bytes;
    std::vector<id<MTLBuffer>> conv_states;
    std::vector<id<MTLBuffer>> conv_bak;
    std::vector<id<MTLBuffer>> ssm_states;
    std::vector<id<MTLBuffer>> ssm_bak;
    std::vector<id<MTLBuffer>> k_caches;
    std::vector<id<MTLBuffer>> v_caches;

    bool ensure_kv_cap(uint32_t needed_seq) {
        if (kv_stride == 0u || needed_seq <= cap_seq) return true;
        uint32_t new_cap = std::max<uint32_t>(cap_seq * 2u, needed_seq + 256u);
        size_t old_bytes = size_t(cap_seq) * size_t(kv_stride) * sizeof(float);
        size_t new_bytes = size_t(new_cap) * size_t(kv_stride) * sizeof(float);
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (is_rec[il]) continue;
            id<MTLBuffer> nk = cb4_alloc_zeroed(device, new_bytes);
            id<MTLBuffer> nv = cb4_alloc_zeroed(device, new_bytes);
            if (!nk || !nv) return false;
            std::memcpy([nk contents], [k_caches[il] contents], old_bytes);
            std::memcpy([nv contents], [v_caches[il] contents], old_bytes);
            k_caches[il] = nk;
            v_caches[il] = nv;
        }
        cap_seq = new_cap;
        return true;
    }
};

extern "C" {

struct VxtCb4FastSubMat {
    uint32_t rows;
    uint32_t cols;
    uint32_t padded_cols;
    uint32_t nnz;
    float poly[4];
    const uint16_t* scales_f16;
    const uint32_t* packed_words;
    const uint32_t* outlier_row_ptr;
    const uint32_t* outliers_packed;
};

struct VxtCb4FastConcatMat {
    uint32_t n_subs;
    VxtCb4FastSubMat subs[3];
};

struct VxtCb4FastLayerUpload {
    uint32_t is_recurrent;
    uint32_t gated_q;
    uint32_t ab_is_strided32;
    uint32_t _pad;
    const float* attn_norm;
    const float* post_attn_norm;
    VxtCb4FastConcatMat w_in;
    VxtCb4FastConcatMat w_out;
    VxtCb4FastConcatMat w_ffn_gu;
    VxtCb4FastConcatMat w_ffn_dn;
    const float* ab_weights_pad;
    const float* vec0; uint32_t vec0_len;
    const float* vec1; uint32_t vec1_len;
    const float* vec2; uint32_t vec2_len;
    const float* vec3; uint32_t vec3_len;
    const float* vec4; uint32_t vec4_len;
};

} // extern "C"
