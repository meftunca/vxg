#include "metal_internal.h"

#include <vector>

namespace {

struct RmsNormParams {
    uint32_t dim;
    float eps;
};

struct DeltaConvParams {
    uint32_t conv_ch;
    uint32_t k_sz;
};

struct DeltaHeadsParams {
    uint32_t n_kh;
    uint32_t n_vh;
    uint32_t dk;
    uint32_t dv;
    float eps;
};

struct GqaParams {
    uint32_t nh;
    uint32_t nkv;
    uint32_t hd;
    uint32_t pos;
    uint32_t rope_dim;
    uint32_t flags;
    float rope_freq_base;
    float eps;
};

id<MTLBuffer> alloc_zeroed_floats(id<MTLDevice> dev, size_t elems) {
    size_t bytes = std::max<size_t>(elems * sizeof(float), 64);
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (b) {
        std::memset([b contents], 0, bytes);
    }
    return b;
}

void retain_op_bufs(std::vector<id<MTLBuffer>>& pool, const VxtMetalOp& op) {
    if (op.b0) pool.push_back((__bridge id<MTLBuffer>)op.b0);
    if (op.b1) pool.push_back((__bridge id<MTLBuffer>)op.b1);
    if (op.b2) pool.push_back((__bridge id<MTLBuffer>)op.b2);
    if (op.b3) pool.push_back((__bridge id<MTLBuffer>)op.b3);
}

struct MetalLlmRunner {
    MetalGemvCtx gemv;
    VxtLlmConfig cfg{};
    std::vector<VxtLlmLayerDesc> layers;
    VxtMetalOp output_head_op{};
    id<MTLBuffer> output_norm_buf = nil;
    std::vector<id<MTLBuffer>> retained_bufs;

    id<MTLComputePipelineState> pso_rms_norm = nil;
    id<MTLComputePipelineState> pso_add_rms_norm = nil;
    id<MTLComputePipelineState> pso_vec_add = nil;
    id<MTLComputePipelineState> pso_swiglu_mul = nil;
    id<MTLComputePipelineState> pso_deltanet_conv = nil;
    id<MTLComputePipelineState> pso_deltanet_heads = nil;
    id<MTLComputePipelineState> pso_gqa_prep = nil;
    id<MTLComputePipelineState> pso_gqa_attn = nil;

    id<MTLBuffer> dummy_buf = nil;
    id<MTLBuffer> cur_buf = nil;
    id<MTLBuffer> normed_buf = nil;
    id<MTLBuffer> x_rot_buf = nil;
    id<MTLBuffer> attn_out_buf = nil;
    id<MTLBuffer> ffn_gate_buf = nil;
    id<MTLBuffer> ffn_up_buf = nil;
    id<MTLBuffer> ffn_out_buf = nil;
    id<MTLBuffer> qkv_mixed_buf = nil;
    id<MTLBuffer> z_buf = nil;
    id<MTLBuffer> alpha_buf = nil;
    id<MTLBuffer> beta_buf = nil;
    id<MTLBuffer> conv_silu_buf = nil;
    id<MTLBuffer> y_concat_buf = nil;
    id<MTLBuffer> q_raw_buf = nil;
    id<MTLBuffer> k_cur_buf = nil;
    id<MTLBuffer> v_cur_buf = nil;
    id<MTLBuffer> q_cur_buf = nil;
    id<MTLBuffer> q_gate_buf = nil;
    id<MTLBuffer> att_cat_buf = nil;
    id<MTLBuffer> logits_buf = nil;
};

struct MetalLlmState {
    id<MTLDevice> device = nil;
    uint32_t n_layer = 0;
    uint32_t kv_stride = 0;
    uint32_t cap_seq = 0;
    std::vector<uint8_t> is_rec;
    std::vector<id<MTLBuffer>> conv_states;
    std::vector<id<MTLBuffer>> ssm_states;
    std::vector<id<MTLBuffer>> k_caches;
    std::vector<id<MTLBuffer>> v_caches;

    bool ensure_kv_cap(uint32_t pos) {
        if (pos < cap_seq || kv_stride == 0) return true;
        uint32_t new_cap = std::max<uint32_t>(cap_seq * 2u, pos + 256u);
        size_t old_bytes = size_t(cap_seq) * size_t(kv_stride) * sizeof(float);
        size_t new_elems = size_t(new_cap) * size_t(kv_stride);
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (is_rec[il]) continue;
            id<MTLBuffer> nk = alloc_zeroed_floats(device, new_elems);
            id<MTLBuffer> nv = alloc_zeroed_floats(device, new_elems);
            if (!nk || !nv) return false;
            if (old_bytes > 0) {
                std::memcpy([nk contents], [k_caches[il] contents], old_bytes);
                std::memcpy([nv contents], [v_caches[il] contents], old_bytes);
            }
            k_caches[il] = nk;
            v_caches[il] = nv;
        }
        cap_seq = new_cap;
        return true;
    }
};

void dispatch_1d(id<MTLComputeCommandEncoder> enc, uint32_t total_threads) {
    uint32_t tg_w = 256u;
    uint32_t groups = (total_threads + tg_w - 1u) / tg_w;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
}

void maybe_rot_ops(
    MetalGemvCtx* g,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> x_in,
    id<MTLBuffer> x_rot,
    const VxtMetalOp* ops,
    uint32_t n
) {
    uint32_t max_cols = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (ops[i].kind == 4u) {
            max_cols = std::max(max_cols, ops[i].cols);
        }
    }
    if (max_cols > 0) {
        vxt_encode_fwht32_buf(g, enc, x_in, x_rot, max_cols);
    }
}

} // namespace

extern "C" {

void* vxt_metal_llm_create(
    void* raw_gemv_ctx,
    const VxtLlmConfig* cfg,
    const VxtLlmLayerDesc* layer_descs,
    void* output_norm_buf,
    const VxtMetalOp* output_head_op,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* src_ctx = static_cast<MetalGemvCtx*>(raw_gemv_ctx);
        if (!src_ctx || !cfg || !layer_descs || !output_norm_buf || !output_head_op) {
            vxt_write_err(err_out, err_cap, "null argument to vxt_metal_llm_create");
            return nullptr;
        }
        uint32_t dv = cfg->ssm_dt_rank > 0 ? (cfg->ssm_inner_size / cfg->ssm_dt_rank) : 0;
        if (cfg->head_dim > 512u || cfg->ssm_state_size > 512u || dv > 512u) {
            vxt_write_err(err_out, err_cap, "head_dim/ssm_dim > 512 unsupported in Metal LLM kernel");
            return nullptr;
        }

        auto* r = new MetalLlmRunner();
        r->gemv = *src_ctx;
        r->cfg = *cfg;
        r->layers.assign(layer_descs, layer_descs + cfg->n_layer);
        r->output_head_op = *output_head_op;
        r->output_norm_buf = (__bridge id<MTLBuffer>)output_norm_buf;

        id<MTLDevice> dev = r->gemv.device;
        id<MTLLibrary> lib = r->gemv.lib;
        r->pso_rms_norm = vxt_build_pso(dev, lib, @"vxt_rms_norm", err_out, err_cap);
        r->pso_add_rms_norm = vxt_build_pso(dev, lib, @"vxt_add_rms_norm", err_out, err_cap);
        r->pso_vec_add = vxt_build_pso(dev, lib, @"vxt_vec_add", err_out, err_cap);
        r->pso_swiglu_mul = vxt_build_pso(dev, lib, @"vxt_swiglu_mul", err_out, err_cap);
        r->pso_deltanet_conv = vxt_build_pso(dev, lib, @"vxt_deltanet_conv", err_out, err_cap);
        r->pso_deltanet_heads = vxt_build_pso(dev, lib, @"vxt_deltanet_heads", err_out, err_cap);
        r->pso_gqa_prep = vxt_build_pso(dev, lib, @"vxt_gqa_prep", err_out, err_cap);
        r->pso_gqa_attn = vxt_build_pso(dev, lib, @"vxt_gqa_attn", err_out, err_cap);
        if (!r->pso_rms_norm || !r->pso_add_rms_norm || !r->pso_vec_add || !r->pso_swiglu_mul
            || !r->pso_deltanet_conv || !r->pso_deltanet_heads || !r->pso_gqa_prep || !r->pso_gqa_attn) {
            delete r;
            return nullptr;
        }

        for (const auto& l : r->layers) {
            if (l.attn_norm_buf) r->retained_bufs.push_back((__bridge id<MTLBuffer>)l.attn_norm_buf);
            if (l.post_attn_norm_buf) r->retained_bufs.push_back((__bridge id<MTLBuffer>)l.post_attn_norm_buf);
            if (l.vec0_buf) r->retained_bufs.push_back((__bridge id<MTLBuffer>)l.vec0_buf);
            if (l.vec1_buf) r->retained_bufs.push_back((__bridge id<MTLBuffer>)l.vec1_buf);
            if (l.vec2_buf) r->retained_bufs.push_back((__bridge id<MTLBuffer>)l.vec2_buf);
            if (l.vec3_buf) r->retained_bufs.push_back((__bridge id<MTLBuffer>)l.vec3_buf);
            if (l.vec4_buf) r->retained_bufs.push_back((__bridge id<MTLBuffer>)l.vec4_buf);
            retain_op_bufs(r->retained_bufs, l.op_a0);
            retain_op_bufs(r->retained_bufs, l.op_a1);
            retain_op_bufs(r->retained_bufs, l.op_a2);
            retain_op_bufs(r->retained_bufs, l.op_a3);
            retain_op_bufs(r->retained_bufs, l.op_out);
            retain_op_bufs(r->retained_bufs, l.op_ffn_gate);
            retain_op_bufs(r->retained_bufs, l.op_ffn_up);
            retain_op_bufs(r->retained_bufs, l.op_ffn_down);
        }
        retain_op_bufs(r->retained_bufs, r->output_head_op);

        size_t conv_ch = size_t(cfg->ssm_inner_size) + 2u * size_t(cfg->ssm_group_count) * size_t(cfg->ssm_state_size);
        size_t q_dim = size_t(cfg->n_head) * size_t(cfg->head_dim);
        size_t kv_dim = size_t(cfg->n_head_kv) * size_t(cfg->head_dim);
        size_t max_qkv = conv_ch;
        size_t max_q_raw = 2u * q_dim;
        for (const auto& l : r->layers) {
            if (l.is_recurrent) max_qkv = std::max(max_qkv, size_t(l.op_a0.rows));
            else max_q_raw = std::max(max_q_raw, size_t(l.op_a0.rows));
        }
        size_t max_act = std::max({size_t(cfg->n_embd), size_t(cfg->n_ff), size_t(cfg->ssm_inner_size), q_dim});

        r->dummy_buf = alloc_zeroed_floats(dev, 16);
        r->cur_buf = alloc_zeroed_floats(dev, size_t(cfg->n_embd) + 512u);
        r->normed_buf = alloc_zeroed_floats(dev, size_t(cfg->n_embd) + 512u);
        r->x_rot_buf = alloc_zeroed_floats(dev, max_act + 512u);
        r->attn_out_buf = alloc_zeroed_floats(dev, size_t(cfg->n_embd) + 512u);
        r->ffn_gate_buf = alloc_zeroed_floats(dev, size_t(cfg->n_ff) + 512u);
        r->ffn_up_buf = alloc_zeroed_floats(dev, size_t(cfg->n_ff) + 512u);
        r->ffn_out_buf = alloc_zeroed_floats(dev, size_t(cfg->n_embd) + 512u);
        r->qkv_mixed_buf = alloc_zeroed_floats(dev, max_qkv + 512u);
        r->z_buf = alloc_zeroed_floats(dev, size_t(cfg->ssm_inner_size) + 512u);
        r->alpha_buf = alloc_zeroed_floats(dev, size_t(cfg->ssm_dt_rank) + 64u);
        r->beta_buf = alloc_zeroed_floats(dev, size_t(cfg->ssm_dt_rank) + 64u);
        r->conv_silu_buf = alloc_zeroed_floats(dev, max_qkv + 64u);
        r->y_concat_buf = alloc_zeroed_floats(dev, size_t(cfg->ssm_inner_size) + 512u);
        r->q_raw_buf = alloc_zeroed_floats(dev, max_q_raw + 512u);
        r->k_cur_buf = alloc_zeroed_floats(dev, kv_dim + 512u);
        r->v_cur_buf = alloc_zeroed_floats(dev, kv_dim + 512u);
        r->q_cur_buf = alloc_zeroed_floats(dev, q_dim + 512u);
        r->q_gate_buf = alloc_zeroed_floats(dev, q_dim + 512u);
        r->att_cat_buf = alloc_zeroed_floats(dev, q_dim + 512u);
        r->logits_buf = alloc_zeroed_floats(dev, size_t(cfg->n_vocab) + 512u);

        if (!r->dummy_buf || !r->cur_buf || !r->normed_buf || !r->x_rot_buf || !r->logits_buf) {
            vxt_write_err(err_out, err_cap, "failed to allocate Metal LLM scratch buffers");
            delete r;
            return nullptr;
        }
        return r;
    }
}

void vxt_metal_llm_destroy(void* raw_runner) {
    if (!raw_runner) return;
    @autoreleasepool {
        delete static_cast<MetalLlmRunner*>(raw_runner);
    }
}

void* vxt_metal_llm_state_create(void* raw_runner, char* err_out, size_t err_cap) {
    @autoreleasepool {
        auto* r = static_cast<MetalLlmRunner*>(raw_runner);
        if (!r) {
            vxt_write_err(err_out, err_cap, "null runner in vxt_metal_llm_state_create");
            return nullptr;
        }
        const auto& cfg = r->cfg;
        auto* st = new MetalLlmState();
        st->device = r->gemv.device;
        st->n_layer = cfg.n_layer;
        st->kv_stride = cfg.n_head_kv * cfg.head_dim;
        st->cap_seq = 512u;
        st->is_rec.resize(cfg.n_layer, 0);
        st->conv_states.resize(cfg.n_layer, nil);
        st->ssm_states.resize(cfg.n_layer, nil);
        st->k_caches.resize(cfg.n_layer, nil);
        st->v_caches.resize(cfg.n_layer, nil);

        size_t conv_ch = size_t(cfg.ssm_inner_size) + 2u * size_t(cfg.ssm_group_count) * size_t(cfg.ssm_state_size);
        size_t conv_hist = (cfg.ssm_conv_kernel > 0 ? size_t(cfg.ssm_conv_kernel - 1u) : 0u) * conv_ch;
        size_t dv = cfg.ssm_dt_rank > 0 ? size_t(cfg.ssm_inner_size / cfg.ssm_dt_rank) : 0u;
        size_t ssm_len = size_t(cfg.ssm_dt_rank) * dv * size_t(cfg.ssm_state_size);

        for (uint32_t il = 0; il < cfg.n_layer; ++il) {
            if (r->layers[il].is_recurrent) {
                st->is_rec[il] = 1;
                st->conv_states[il] = alloc_zeroed_floats(st->device, conv_hist);
                st->ssm_states[il] = alloc_zeroed_floats(st->device, ssm_len);
                if (!st->conv_states[il] || !st->ssm_states[il]) {
                    vxt_write_err(err_out, err_cap, "failed to allocate Metal DeltaNet state");
                    delete st;
                    return nullptr;
                }
            } else {
                size_t kv_elems = size_t(st->cap_seq) * size_t(st->kv_stride);
                st->k_caches[il] = alloc_zeroed_floats(st->device, kv_elems);
                st->v_caches[il] = alloc_zeroed_floats(st->device, kv_elems);
                if (!st->k_caches[il] || !st->v_caches[il]) {
                    vxt_write_err(err_out, err_cap, "failed to allocate Metal KV cache");
                    delete st;
                    return nullptr;
                }
            }
        }
        return st;
    }
}

void vxt_metal_llm_state_destroy(void* raw_state) {
    if (!raw_state) return;
    @autoreleasepool {
        delete static_cast<MetalLlmState*>(raw_state);
    }
}

int vxt_metal_llm_step(
    void* raw_runner,
    void* raw_state,
    const float* cur_init,
    uint32_t pos,
    uint32_t compute_logits,
    float* logits_out,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* r = static_cast<MetalLlmRunner*>(raw_runner);
        auto* st = static_cast<MetalLlmState*>(raw_state);
        if (!r || !st || !cur_init || (compute_logits && !logits_out)) {
            vxt_write_err(err_out, err_cap, "invalid arguments to vxt_metal_llm_step");
            return -1;
        }
        if (!st->ensure_kv_cap(pos)) {
            vxt_write_err(err_out, err_cap, "KV cache growth failed");
            return -1;
        }

        const auto& cfg = r->cfg;
        uint32_t dim = cfg.n_embd;
        uint32_t n_ff = cfg.n_ff;
        RmsNormParams norm_p{dim, cfg.rms_norm_eps};
        MTLSize tg256 = MTLSizeMake(256, 1, 1);
        MTLSize one_tg = MTLSizeMake(1, 1, 1);

        std::memcpy([r->cur_buf contents], cur_init, size_t(dim) * sizeof(float));

        id<MTLCommandBuffer> cmd = [r->gemv.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

        uint32_t conv_ch = cfg.ssm_inner_size + 2u * cfg.ssm_group_count * cfg.ssm_state_size;
        uint32_t dv = cfg.ssm_dt_rank > 0 ? (cfg.ssm_inner_size / cfg.ssm_dt_rank) : 0u;
        DeltaConvParams dconv_p{conv_ch, cfg.ssm_conv_kernel};
        DeltaHeadsParams dhead_p{
            cfg.ssm_group_count, cfg.ssm_dt_rank, cfg.ssm_state_size, dv, cfg.rms_norm_eps
        };

        for (uint32_t il = 0; il < cfg.n_layer; ++il) {
            const auto& ly = r->layers[il];
            if (il > 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:r->pso_rms_norm];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)ly.attn_norm_buf offset:0 atIndex:1];
            [enc setBuffer:r->normed_buf offset:0 atIndex:2];
            [enc setBytes:&norm_p length:sizeof(norm_p) atIndex:3];
            [enc dispatchThreadgroups:one_tg threadsPerThreadgroup:tg256];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            if (ly.is_recurrent) {
                VxtMetalOp a_ops[4] = {ly.op_a0, ly.op_a1, ly.op_a2, ly.op_a3};
                maybe_rot_ops(&r->gemv, enc, r->normed_buf, r->x_rot_buf, a_ops, 4);
                vxt_encode_gemv_op(&r->gemv, enc, ly.op_a0, r->normed_buf, r->x_rot_buf, r->qkv_mixed_buf, 0);
                vxt_encode_gemv_op(&r->gemv, enc, ly.op_a1, r->normed_buf, r->x_rot_buf, r->z_buf, 0);
                vxt_encode_gemv_op(&r->gemv, enc, ly.op_a2, r->normed_buf, r->x_rot_buf, r->alpha_buf, 0);
                vxt_encode_gemv_op(&r->gemv, enc, ly.op_a3, r->normed_buf, r->x_rot_buf, r->beta_buf, 0);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                [enc setComputePipelineState:r->pso_deltanet_conv];
                [enc setBuffer:r->qkv_mixed_buf offset:0 atIndex:0];
                [enc setBuffer:(__bridge id<MTLBuffer>)ly.vec0_buf offset:0 atIndex:1];
                [enc setBuffer:st->conv_states[il] offset:0 atIndex:2];
                [enc setBuffer:r->conv_silu_buf offset:0 atIndex:3];
                [enc setBytes:&dconv_p length:sizeof(dconv_p) atIndex:4];
                dispatch_1d(enc, conv_ch);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                [enc setComputePipelineState:r->pso_deltanet_heads];
                [enc setBuffer:r->conv_silu_buf offset:0 atIndex:0];
                [enc setBuffer:r->z_buf offset:0 atIndex:1];
                [enc setBuffer:r->alpha_buf offset:0 atIndex:2];
                [enc setBuffer:r->beta_buf offset:0 atIndex:3];
                [enc setBuffer:(__bridge id<MTLBuffer>)ly.vec1_buf offset:0 atIndex:4];
                [enc setBuffer:(__bridge id<MTLBuffer>)ly.vec2_buf offset:0 atIndex:5];
                [enc setBuffer:(__bridge id<MTLBuffer>)ly.vec3_buf offset:0 atIndex:6];
                [enc setBuffer:st->ssm_states[il] offset:0 atIndex:7];
                [enc setBuffer:r->y_concat_buf offset:0 atIndex:8];
                [enc setBytes:&dhead_p length:sizeof(dhead_p) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.ssm_dt_rank, 1, 1) threadsPerThreadgroup:tg256];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                maybe_rot_ops(&r->gemv, enc, r->y_concat_buf, r->x_rot_buf, &ly.op_out, 1);
                vxt_encode_gemv_op(&r->gemv, enc, ly.op_out, r->y_concat_buf, r->x_rot_buf, r->attn_out_buf, 0);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else {
                VxtMetalOp g_ops[3] = {ly.op_a0, ly.op_a1, ly.op_a2};
                maybe_rot_ops(&r->gemv, enc, r->normed_buf, r->x_rot_buf, g_ops, 3);
                vxt_encode_gemv_op(&r->gemv, enc, ly.op_a0, r->normed_buf, r->x_rot_buf, r->q_raw_buf, 0);
                vxt_encode_gemv_op(&r->gemv, enc, ly.op_a1, r->normed_buf, r->x_rot_buf, r->k_cur_buf, 0);
                vxt_encode_gemv_op(&r->gemv, enc, ly.op_a2, r->normed_buf, r->x_rot_buf, r->v_cur_buf, 0);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                uint32_t flags = (ly.gated_q ? 1u : 0u)
                    | (cfg.rope_neox ? 2u : 0u)
                    | (ly.vec0_buf ? 4u : 0u)
                    | (ly.vec1_buf ? 8u : 0u)
                    | (ly.vec2_buf ? 16u : 0u)
                    | (ly.vec3_buf ? 32u : 0u)
                    | (ly.vec4_buf ? 64u : 0u);
                GqaParams gp{
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim, pos,
                    cfg.rope_dim, flags, cfg.rope_freq_base, cfg.rms_norm_eps
                };
                auto b_or_dummy = [&](void* p) -> id<MTLBuffer> {
                    return p ? (__bridge id<MTLBuffer>)p : r->dummy_buf;
                };

                [enc setComputePipelineState:r->pso_gqa_prep];
                [enc setBuffer:r->q_raw_buf offset:0 atIndex:0];
                [enc setBuffer:r->k_cur_buf offset:0 atIndex:1];
                [enc setBuffer:r->v_cur_buf offset:0 atIndex:2];
                [enc setBuffer:b_or_dummy(ly.vec0_buf) offset:0 atIndex:3];
                [enc setBuffer:b_or_dummy(ly.vec1_buf) offset:0 atIndex:4];
                [enc setBuffer:b_or_dummy(ly.vec2_buf) offset:0 atIndex:5];
                [enc setBuffer:b_or_dummy(ly.vec3_buf) offset:0 atIndex:6];
                [enc setBuffer:b_or_dummy(ly.vec4_buf) offset:0 atIndex:7];
                [enc setBuffer:r->q_cur_buf offset:0 atIndex:8];
                [enc setBuffer:r->q_gate_buf offset:0 atIndex:9];
                [enc setBuffer:st->k_caches[il] offset:0 atIndex:10];
                [enc setBuffer:st->v_caches[il] offset:0 atIndex:11];
                [enc setBytes:&gp length:sizeof(gp) atIndex:12];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.n_head + cfg.n_head_kv, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                [enc setComputePipelineState:r->pso_gqa_attn];
                [enc setBuffer:r->q_cur_buf offset:0 atIndex:0];
                [enc setBuffer:r->q_gate_buf offset:0 atIndex:1];
                [enc setBuffer:st->k_caches[il] offset:0 atIndex:2];
                [enc setBuffer:st->v_caches[il] offset:0 atIndex:3];
                [enc setBuffer:r->att_cat_buf offset:0 atIndex:4];
                [enc setBytes:&gp length:sizeof(gp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(cfg.n_head, 1, 1) threadsPerThreadgroup:tg256];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                maybe_rot_ops(&r->gemv, enc, r->att_cat_buf, r->x_rot_buf, &ly.op_out, 1);
                vxt_encode_gemv_op(&r->gemv, enc, ly.op_out, r->att_cat_buf, r->x_rot_buf, r->attn_out_buf, 0);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

            [enc setComputePipelineState:r->pso_add_rms_norm];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->attn_out_buf offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)ly.post_attn_norm_buf offset:0 atIndex:2];
            [enc setBuffer:r->normed_buf offset:0 atIndex:3];
            [enc setBytes:&norm_p length:sizeof(norm_p) atIndex:4];
            [enc dispatchThreadgroups:one_tg threadsPerThreadgroup:tg256];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            VxtMetalOp ffn_ops[2] = {ly.op_ffn_gate, ly.op_ffn_up};
            maybe_rot_ops(&r->gemv, enc, r->normed_buf, r->x_rot_buf, ffn_ops, 2);
            vxt_encode_gemv_op(&r->gemv, enc, ly.op_ffn_gate, r->normed_buf, r->x_rot_buf, r->ffn_gate_buf, 0);
            vxt_encode_gemv_op(&r->gemv, enc, ly.op_ffn_up, r->normed_buf, r->x_rot_buf, r->ffn_up_buf, 0);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:r->pso_swiglu_mul];
            [enc setBuffer:r->ffn_gate_buf offset:0 atIndex:0];
            [enc setBuffer:r->ffn_up_buf offset:0 atIndex:1];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:2];
            dispatch_1d(enc, n_ff);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            maybe_rot_ops(&r->gemv, enc, r->ffn_gate_buf, r->x_rot_buf, &ly.op_ffn_down, 1);
            vxt_encode_gemv_op(&r->gemv, enc, ly.op_ffn_down, r->ffn_gate_buf, r->x_rot_buf, r->ffn_out_buf, 0);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:r->pso_vec_add];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->ffn_out_buf offset:0 atIndex:1];
            [enc setBytes:&dim length:sizeof(dim) atIndex:2];
            dispatch_1d(enc, dim);
        }

        if (compute_logits) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:r->pso_rms_norm];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->output_norm_buf offset:0 atIndex:1];
            [enc setBuffer:r->normed_buf offset:0 atIndex:2];
            [enc setBytes:&norm_p length:sizeof(norm_p) atIndex:3];
            [enc dispatchThreadgroups:one_tg threadsPerThreadgroup:tg256];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            maybe_rot_ops(&r->gemv, enc, r->normed_buf, r->x_rot_buf, &r->output_head_op, 1);
            vxt_encode_gemv_op(&r->gemv, enc, r->output_head_op, r->normed_buf, r->x_rot_buf, r->logits_buf, 0);
        }

        [enc endEncoding];
        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        [cmd commit];
        while ([cmd status] < MTLCommandBufferStatusCompleted) {
            if (CFAbsoluteTimeGetCurrent() - t0 > 0.015) {
                [cmd waitUntilCompleted];
                break;
            }
#if defined(__aarch64__)
            __builtin_arm_yield();
#endif
        }
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            NSError* e = [cmd error];
            vxt_write_err(err_out, err_cap, e ? [[e localizedDescription] UTF8String] : "LLM MTLCommandBuffer failed");
            return -1;
        }
        if (compute_logits) {
            std::memcpy(logits_out, [r->logits_buf contents], size_t(cfg.n_vocab) * sizeof(float));
        }
        return 0;
    }
}

} // extern "C"
