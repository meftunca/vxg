#include "metal_internal.h"
#include <cmath>
#include <vector>

namespace {

static inline size_t vis_align256(size_t n) {
    return (std::max<size_t>(n, 16) + 255u) & ~size_t(255u);
}

struct VisNormParamsC { uint32_t rows, dim; float eps; uint32_t _pad; };
struct VisGemmParamsC { uint32_t n_rows, m_dim, k_dim, act_mode; };
struct VisRopeParamsC { uint32_t seq, nh, hd, spatial_dim; };
struct VisMhaParamsC  { uint32_t seq, nh, hd; float scale; };

struct VisBlockOffs {
    size_t norm1_w, norm1_b;
    size_t qkv_w, qkv_b;
    size_t proj_w, proj_b;
    size_t norm2_w, norm2_b;
    size_t fc1_w, fc1_b;
    size_t fc2_w, fc2_b;
};

struct MetalVisionRunnerC {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    uint32_t depth = 0;
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;
    uint32_t patch_dim = 0;
    uint32_t merge_unit = 0;
    uint32_t out_hidden_size = 0;

    id<MTLComputePipelineState> pso_add = nil;
    id<MTLComputePipelineState> pso_ln = nil;
    id<MTLComputePipelineState> pso_gemm = nil;
    id<MTLComputePipelineState> pso_rope2d = nil;
    id<MTLComputePipelineState> pso_mha = nil;

    id<MTLBuffer> arena_buf = nil;
    size_t patch_w_off = 0, patch_b_off = 0;
    std::vector<VisBlockOffs> blocks;
    size_t m_norm_w_off = 0, m_norm_b_off = 0;
    size_t m_fc1_w_off = 0, m_fc1_b_off = 0;
    size_t m_fc2_w_off = 0, m_fc2_b_off = 0;

    uint32_t cap_seq = 0;
    id<MTLBuffer> patches_buf = nil;
    id<MTLBuffer> pos_add_buf = nil;
    id<MTLBuffer> rope_cos_buf = nil;
    id<MTLBuffer> rope_sin_buf = nil;
    id<MTLBuffer> hidden_buf = nil;
    id<MTLBuffer> normed_buf = nil;
    id<MTLBuffer> qkv_buf = nil;
    id<MTLBuffer> attn_cat_buf = nil;
    id<MTLBuffer> fc1_buf = nil;
    id<MTLBuffer> m_fc1_buf = nil;
    id<MTLBuffer> embeds_buf = nil;

    bool ensure_seq(uint32_t seq) {
        if (seq <= cap_seq && hidden_buf != nil) return true;
        uint32_t nc = std::max<uint32_t>(cap_seq * 2u, std::max<uint32_t>(seq, 64u));
        uint32_t llm_toks = std::max<uint32_t>(nc / std::max<uint32_t>(merge_unit, 1u), 1u);
        size_t mdim = size_t(hidden_size) * size_t(merge_unit);
        auto alloc = [&](size_t elems) -> id<MTLBuffer> {
            return [device newBufferWithLength:std::max<size_t>(elems * sizeof(float), 64u)
                                       options:MTLResourceStorageModeShared];
        };
        uint32_t sp_dim = head_dim / 2u;
        patches_buf  = alloc(size_t(nc) * patch_dim);
        pos_add_buf  = alloc(size_t(nc) * hidden_size);
        rope_cos_buf = alloc(size_t(nc) * sp_dim);
        rope_sin_buf = alloc(size_t(nc) * sp_dim);
        hidden_buf   = alloc(size_t(nc) * hidden_size);
        normed_buf   = alloc(size_t(nc) * hidden_size);
        qkv_buf      = alloc(size_t(nc) * 3u * hidden_size);
        attn_cat_buf = alloc(size_t(nc) * hidden_size);
        fc1_buf      = alloc(size_t(nc) * intermediate_size);
        m_fc1_buf    = alloc(size_t(llm_toks) * mdim);
        embeds_buf   = alloc(size_t(llm_toks) * out_hidden_size);
        if (!patches_buf || !pos_add_buf || !rope_cos_buf || !rope_sin_buf || !hidden_buf ||
            !normed_buf || !qkv_buf || !attn_cat_buf || !fc1_buf || !m_fc1_buf || !embeds_buf) {
            return false;
        }
        cap_seq = nc;
        return true;
    }
};

} // namespace

extern "C" {

struct VxtVisionBlockUpload {
    const float* norm1_w;
    const float* norm1_b;
    const uint16_t* qkv_w_f16;
    const float* qkv_b;
    const uint16_t* proj_w_f16;
    const float* proj_b;
    const float* norm2_w;
    const float* norm2_b;
    const uint16_t* fc1_w_f16;
    const float* fc1_b;
    const uint16_t* fc2_w_f16;
    const float* fc2_b;
};

struct VxtVisionModelUpload {
    uint32_t depth;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t patch_dim;
    uint32_t merge_unit;
    uint32_t out_hidden_size;
    const uint16_t* patch_w_f16;
    const float* patch_b;
    const VxtVisionBlockUpload* blocks;
    const float* merger_norm_w;
    const float* merger_norm_b;
    const uint16_t* merger_fc1_w_f16;
    const float* merger_fc1_b;
    const uint16_t* merger_fc2_w_f16;
    const float* merger_fc2_b;
};

void* vxt_metal_vision_create(
    void* gemv_ctx_raw,
    const VxtVisionModelUpload* up,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* ctx = static_cast<MetalGemvCtx*>(gemv_ctx_raw);
        if (!ctx || !ctx->device || !ctx->queue || !ctx->lib || !up || up->head_dim > 256u) {
            vxt_write_err(err_out, err_cap, "invalid arguments to vxt_metal_vision_create");
            return nullptr;
        }
        auto* r = new MetalVisionRunnerC();
        r->device = ctx->device;
        r->queue = ctx->queue;
        r->depth = up->depth;
        r->hidden_size = up->hidden_size;
        r->intermediate_size = up->intermediate_size;
        r->num_heads = up->num_heads;
        r->head_dim = up->head_dim;
        r->patch_dim = up->patch_dim;
        r->merge_unit = up->merge_unit;
        r->out_hidden_size = up->out_hidden_size;

        r->pso_add    = vxt_build_pso(r->device, ctx->lib, @"vxt_vis_add_inplace", err_out, err_cap);
        r->pso_ln     = vxt_build_pso(r->device, ctx->lib, @"vxt_vis_layernorm", err_out, err_cap);
        r->pso_gemm   = vxt_build_pso(r->device, ctx->lib, @"vxt_vis_gemm_f16", err_out, err_cap);
        r->pso_rope2d = vxt_build_pso(r->device, ctx->lib, @"vxt_vis_rope2d", err_out, err_cap);
        r->pso_mha    = vxt_build_pso(r->device, ctx->lib, @"vxt_vis_mha", err_out, err_cap);
        if (!r->pso_add || !r->pso_ln || !r->pso_gemm || !r->pso_rope2d || !r->pso_mha) {
            delete r;
            return nullptr;
        }

        size_t d = r->hidden_size, ff = r->intermediate_size, pdim = r->patch_dim;
        size_t mdim = d * r->merge_unit, odim = r->out_hidden_size;
        size_t cur = 0u;
        auto reserve = [&](size_t bytes) -> size_t {
            size_t off = cur;
            cur += vis_align256(bytes);
            return off;
        };

        r->patch_w_off = reserve(d * pdim * sizeof(uint16_t));
        r->patch_b_off = reserve(d * sizeof(float));
        r->blocks.resize(r->depth);
        for (uint32_t i = 0; i < r->depth; ++i) {
            auto& b = r->blocks[i];
            b.norm1_w = reserve(d * sizeof(float));
            b.norm1_b = reserve(d * sizeof(float));
            b.qkv_w   = reserve(3u * d * d * sizeof(uint16_t));
            b.qkv_b   = reserve(3u * d * sizeof(float));
            b.proj_w  = reserve(d * d * sizeof(uint16_t));
            b.proj_b  = reserve(d * sizeof(float));
            b.norm2_w = reserve(d * sizeof(float));
            b.norm2_b = reserve(d * sizeof(float));
            b.fc1_w   = reserve(ff * d * sizeof(uint16_t));
            b.fc1_b   = reserve(ff * sizeof(float));
            b.fc2_w   = reserve(d * ff * sizeof(uint16_t));
            b.fc2_b   = reserve(d * sizeof(float));
        }
        r->m_norm_w_off = reserve(d * sizeof(float));
        r->m_norm_b_off = reserve(d * sizeof(float));
        r->m_fc1_w_off  = reserve(mdim * mdim * sizeof(uint16_t));
        r->m_fc1_b_off  = reserve(mdim * sizeof(float));
        r->m_fc2_w_off  = reserve(odim * mdim * sizeof(uint16_t));
        r->m_fc2_b_off  = reserve(odim * sizeof(float));

        r->arena_buf = [r->device newBufferWithLength:std::max<size_t>(cur, 64u)
                                              options:MTLResourceStorageModeShared];
        if (!r->arena_buf) {
            vxt_write_err(err_out, err_cap, "failed to allocate Metal vision weight arena");
            delete r;
            return nullptr;
        }
        auto* base = static_cast<uint8_t*>([r->arena_buf contents]);
        auto cp = [&](size_t off, const void* src, size_t bytes) {
            std::memcpy(base + off, src, bytes);
        };

        cp(r->patch_w_off, up->patch_w_f16, d * pdim * sizeof(uint16_t));
        cp(r->patch_b_off, up->patch_b, d * sizeof(float));
        for (uint32_t i = 0; i < r->depth; ++i) {
            const auto& sb = up->blocks[i];
            const auto& db = r->blocks[i];
            cp(db.norm1_w, sb.norm1_w, d * sizeof(float));
            cp(db.norm1_b, sb.norm1_b, d * sizeof(float));
            cp(db.qkv_w,   sb.qkv_w_f16, 3u * d * d * sizeof(uint16_t));
            cp(db.qkv_b,   sb.qkv_b, 3u * d * sizeof(float));
            cp(db.proj_w,  sb.proj_w_f16, d * d * sizeof(uint16_t));
            cp(db.proj_b,  sb.proj_b, d * sizeof(float));
            cp(db.norm2_w, sb.norm2_w, d * sizeof(float));
            cp(db.norm2_b, sb.norm2_b, d * sizeof(float));
            cp(db.fc1_w,   sb.fc1_w_f16, ff * d * sizeof(uint16_t));
            cp(db.fc1_b,   sb.fc1_b, ff * sizeof(float));
            cp(db.fc2_w,   sb.fc2_w_f16, d * ff * sizeof(uint16_t));
            cp(db.fc2_b,   sb.fc2_b, d * sizeof(float));
        }
        cp(r->m_norm_w_off, up->merger_norm_w, d * sizeof(float));
        cp(r->m_norm_b_off, up->merger_norm_b, d * sizeof(float));
        cp(r->m_fc1_w_off,  up->merger_fc1_w_f16, mdim * mdim * sizeof(uint16_t));
        cp(r->m_fc1_b_off,  up->merger_fc1_b, mdim * sizeof(float));
        cp(r->m_fc2_w_off,  up->merger_fc2_w_f16, odim * mdim * sizeof(uint16_t));
        cp(r->m_fc2_b_off,  up->merger_fc2_b, odim * sizeof(float));
        return r;
    }
}

void vxt_metal_vision_destroy(void* raw_runner) {
    @autoreleasepool {
        delete static_cast<MetalVisionRunnerC*>(raw_runner);
    }
}

int vxt_metal_vision_encode(
    void* raw_runner,
    uint32_t seq,
    const float* patches_in,
    const float* pos_add_in,
    const float* rope_cos_in,
    const float* rope_sin_in,
    float* embeds_out,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* r = static_cast<MetalVisionRunnerC*>(raw_runner);
        if (!r || seq == 0u || (seq % r->merge_unit) != 0u || !patches_in || !pos_add_in ||
            !rope_cos_in || !rope_sin_in || !embeds_out) {
            vxt_write_err(err_out, err_cap, "invalid arguments to vxt_metal_vision_encode");
            return -1;
        }
        if (!r->ensure_seq(seq)) {
            vxt_write_err(err_out, err_cap, "failed to allocate Metal vision workspace");
            return -2;
        }

        uint32_t d = r->hidden_size, ff = r->intermediate_size;
        uint32_t sp_dim = r->head_dim / 2u;
        uint32_t llm_toks = seq / r->merge_unit;
        uint32_t mdim = d * r->merge_unit;

        std::memcpy([r->patches_buf contents], patches_in, size_t(seq) * r->patch_dim * sizeof(float));
        std::memcpy([r->pos_add_buf contents], pos_add_in, size_t(seq) * d * sizeof(float));
        std::memcpy([r->rope_cos_buf contents], rope_cos_in, size_t(seq) * sp_dim * sizeof(float));
        std::memcpy([r->rope_sin_buf contents], rope_sin_in, size_t(seq) * sp_dim * sizeof(float));

        id<MTLCommandBuffer> cmd = [r->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            vxt_write_err(err_out, err_cap, "failed to create Metal vision command buffer");
            return -3;
        }

        MTLSize tg256 = MTLSizeMake(256, 1, 1);
        auto encode_ln = [&](id<MTLBuffer> x, size_t w_off, size_t b_off, id<MTLBuffer> out) {
            VisNormParamsC np{seq, d, 1e-6f, 0u};
            [enc setComputePipelineState:r->pso_ln];
            [enc setBuffer:x offset:0 atIndex:0];
            [enc setBuffer:r->arena_buf offset:w_off atIndex:1];
            [enc setBuffer:r->arena_buf offset:b_off atIndex:2];
            [enc setBuffer:out offset:0 atIndex:3];
            [enc setBytes:&np length:sizeof(np) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(seq, 1, 1) threadsPerThreadgroup:tg256];
        };
        auto encode_gemm = [&](id<MTLBuffer> x, size_t w_off, size_t b_off, id<MTLBuffer> y,
                               uint32_t n_rows, uint32_t m_out, uint32_t k_in, uint32_t act) {
            VisGemmParamsC gp{n_rows, m_out, k_in, act};
            [enc setComputePipelineState:r->pso_gemm];
            [enc setBuffer:x offset:0 atIndex:0];
            [enc setBuffer:r->arena_buf offset:w_off atIndex:1];
            [enc setBuffer:r->arena_buf offset:b_off atIndex:2];
            [enc setBuffer:y offset:0 atIndex:3];
            [enc setBytes:&gp length:sizeof(gp) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((m_out + 7u) / 8u, (n_rows + 7u) / 8u, 1)
                threadsPerThreadgroup:tg256];
        };

        // 1. Patch projection + bilinear pos_embed add
        encode_gemm(r->patches_buf, r->patch_w_off, r->patch_b_off, r->hidden_buf, seq, d, r->patch_dim, 0u);
        uint32_t hid_elems = seq * d;
        [enc setComputePipelineState:r->pso_add];
        [enc setBuffer:r->hidden_buf offset:0 atIndex:0];
        [enc setBuffer:r->pos_add_buf offset:0 atIndex:1];
        [enc setBytes:&hid_elems length:sizeof(hid_elems) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((hid_elems + 255u) / 256u, 1, 1) threadsPerThreadgroup:tg256];

        // 2. ViT transformer blocks
        VisRopeParamsC rp{seq, r->num_heads, r->head_dim, sp_dim};
        VisMhaParamsC mp{seq, r->num_heads, r->head_dim, 1.0f / std::sqrt(float(r->head_dim))};
        uint32_t rope_threads_x = 2u * r->num_heads * sp_dim;

        for (uint32_t i = 0; i < r->depth; ++i) {
            const auto& b = r->blocks[i];
            encode_ln(r->hidden_buf, b.norm1_w, b.norm1_b, r->normed_buf);
            encode_gemm(r->normed_buf, b.qkv_w, b.qkv_b, r->qkv_buf, seq, 3u * d, d, 0u);

            [enc setComputePipelineState:r->pso_rope2d];
            [enc setBuffer:r->qkv_buf offset:0 atIndex:0];
            [enc setBuffer:r->rope_cos_buf offset:0 atIndex:1];
            [enc setBuffer:r->rope_sin_buf offset:0 atIndex:2];
            [enc setBytes:&rp length:sizeof(rp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((rope_threads_x + 63u) / 64u, seq, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            [enc setComputePipelineState:r->pso_mha];
            [enc setBuffer:r->qkv_buf offset:0 atIndex:0];
            [enc setBuffer:r->attn_cat_buf offset:0 atIndex:1];
            [enc setBytes:&mp length:sizeof(mp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(r->num_heads, seq, 1) threadsPerThreadgroup:tg256];

            encode_gemm(r->attn_cat_buf, b.proj_w, b.proj_b, r->hidden_buf, seq, d, d, 1u);
            encode_ln(r->hidden_buf, b.norm2_w, b.norm2_b, r->normed_buf);
            encode_gemm(r->normed_buf, b.fc1_w, b.fc1_b, r->fc1_buf, seq, ff, d, 2u);
            encode_gemm(r->fc1_buf, b.fc2_w, b.fc2_b, r->hidden_buf, seq, d, ff, 1u);
        }

        // 3. 2x2 Spatial PatchMerger (`LayerNorm` -> reshape `[llm_toks, mdim]` -> `fc1 + gelu_exact` -> `fc2`)
        encode_ln(r->hidden_buf, r->m_norm_w_off, r->m_norm_b_off, r->normed_buf);
        encode_gemm(r->normed_buf, r->m_fc1_w_off, r->m_fc1_b_off, r->m_fc1_buf, llm_toks, mdim, mdim, 3u);
        encode_gemm(r->m_fc1_buf, r->m_fc2_w_off, r->m_fc2_b_off, r->embeds_buf, llm_toks, r->out_hidden_size, mdim, 0u);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            NSString* desc = [[cmd error] localizedDescription];
            vxt_write_err(err_out, err_cap, desc ? [desc UTF8String] : "Metal vision command buffer failed");
            return -4;
        }
        std::memcpy(embeds_out, [r->embeds_buf contents], size_t(llm_toks) * r->out_hidden_size * sizeof(float));
        return 0;
    }
}

} // extern "C"
