#include "metal_cb4_fast.h"
#include <cstdio>

namespace {

struct Cb4FastEmbedParamsC {
    uint32_t cols, padded_cols, batch_m, n_vocab;
    float eps, _pad0, _pad1, _pad2;
    float centroids[16];
};
struct Cb4FastNormParamsC { uint32_t dim, padded_dim, batch_m; float eps; };
struct Cb4FastSwigluParamsC { uint32_t n_ff, gu_stride, padded_ff, batch_m; };
struct Cb4FastAbParamsC {
    uint32_t rows, padded_cols, x_stride, y_stride, ab_half_stride, n_vh, batch_m, is_strided32;
};
struct Cb4FastDnetM1ParamsC {
    uint32_t n_kh, n_vh, dk, dv, conv_ch, k_sz, conv_tgs, padded_cols, ab_half_stride, padded_inner, is_strided32;
    float eps;
};
struct Cb4FastDnetBatchParamsC {
    uint32_t n_kh, n_vh, dk, dv, conv_ch, k_sz, batch_m, qkv_z_stride, conv_stride, ab_stride, ab_half_stride, padded_inner;
    float eps, _pad0, _pad1, _pad2;
};
struct Cb4FastGqaBatchParamsC {
    uint32_t nh, nkv, hd, pos_base, rope_dim, flags, batch_m, qkv_z_stride, q_rows, kv_dim, padded_att;
    float rope_freq_base, eps, _pad0, _pad1, _pad2;
};
struct Cb4FastGemvParamsC {
    uint32_t rows, padded_cols, split0, split1, x_stride, y_stride, batch_m, flags;
    float poly[3][4];
};
struct Cb4FastFfnGuParamsC {
    uint32_t n_ff, padded_ff, padded_cols, _pad;
    float poly_g[4], poly_u[4];
};
struct Cb4FastArgmaxParamsC { uint32_t n_vocab, slot, next_slot, num_tgs; };

static void encode_cb4_mat(
    MetalCb4FastRunner* r, id<MTLComputeCommandEncoder> enc, const Cb4ArenaMat& mat,
    id<MTLBuffer> x_buf, size_t x_byte_off, uint32_t x_stride,
    id<MTLBuffer> y_buf, size_t y_byte_off, uint32_t y_stride, uint32_t batch_m, uint32_t add_res
) {
    Cb4FastGemvParamsC p{};
    p.rows = mat.rows; p.padded_cols = mat.padded_cols;
    p.split0 = mat.split0; p.split1 = mat.split1;
    p.x_stride = x_stride; p.y_stride = y_stride;
    p.batch_m = batch_m; p.flags = (add_res != 0u) ? 1u : 0u;
    std::memcpy(p.poly, mat.poly, sizeof(p.poly));

    if (batch_m > 1u) {
        uint32_t tg_threads = 64u, rows_per_tg = (tg_threads >> 5u) << 1u;
        uint32_t num_tgs = (mat.rows + rows_per_tg - 1u) / rows_per_tg;
        uint32_t m_slices = (batch_m + 7u) >> 3u;
        [enc setComputePipelineState:r->pso_gemm_nr2];
        [enc setBuffer:r->weight_arena_buf offset:mat.sc_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:mat.pw_off atIndex:1];
        [enc setBuffer:r->weight_arena_buf offset:mat.rp_off atIndex:2];
        [enc setBuffer:r->weight_arena_buf offset:mat.out_off atIndex:3];
        [enc setBuffer:x_buf offset:x_byte_off atIndex:4];
        [enc setBuffer:y_buf offset:y_byte_off atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, m_slices, 1) threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
        return;
    }

    uint32_t tg_threads = (r->dispatch_mode == 4u) ? 128u : 64u;
    uint32_t rows_per_tg = (tg_threads >> 5u) << (mat.is_nr4 ? 2u : 1u);
    uint32_t num_tgs = (mat.rows + rows_per_tg - 1u) / rows_per_tg;
    [enc setComputePipelineState:(mat.is_nr4 ? r->pso_gemv_nr4 : r->pso_gemv_nr2)];
    [enc setBuffer:r->weight_arena_buf offset:mat.sc_off atIndex:0];
    [enc setBuffer:r->weight_arena_buf offset:mat.pw_off atIndex:1];
    [enc setBuffer:r->weight_arena_buf offset:mat.rp_off atIndex:2];
    [enc setBuffer:r->weight_arena_buf offset:mat.out_off atIndex:3];
    [enc setBuffer:x_buf offset:x_byte_off atIndex:4];
    [enc setBuffer:y_buf offset:y_byte_off atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
}

static void encode_add_rms_norm(
    MetalCb4FastRunner* r, id<MTLComputeCommandEncoder> enc, size_t norm_w_off, uint32_t batch_m
) {
    Cb4FastNormParamsC np{r->cfg.n_embd, r->padded_embd, batch_m, r->cfg.rms_norm_eps};
    [enc setComputePipelineState:r->pso_add_rms_norm];
    [enc setBuffer:r->cur_buf offset:0 atIndex:0];
    [enc setBuffer:r->delta_buf offset:0 atIndex:1];
    [enc setBuffer:r->weight_arena_buf offset:norm_w_off atIndex:2];
    [enc setBuffer:r->normed_buf offset:0 atIndex:3];
    [enc setBytes:&np length:sizeof(np) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, batch_m, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_cb4_fast_forward(
    MetalCb4FastRunner* r, MetalCb4FastState* s, id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> tok_buf, size_t tok_byte_off, uint32_t batch_m,
    uint32_t pos_base, uint32_t compute_mode, uint32_t slot, uint32_t next_slot
) {
    const VxtLlmConfig& c = r->cfg;
    MTLSize tg256 = MTLSizeMake(256, 1, 1);
    bool fuse_m1 = (batch_m == 1u) && !r->disable_fusions;
    bool can_fuse_gu = fuse_m1 && ((c.n_ff & 1u) == 0u);

    Cb4FastEmbedParamsC ep{};
    ep.cols = c.n_embd; ep.padded_cols = r->padded_embd;
    ep.batch_m = batch_m; ep.n_vocab = c.n_vocab; ep.eps = c.rms_norm_eps;
    std::memcpy(ep.centroids, r->embed_centroids, sizeof(ep.centroids));

    bool norm_ready = false;
    if (fuse_m1 && c.n_layer > 0u && r->padded_embd <= 4096u) {
        [enc setComputePipelineState:r->pso_embed_norm0];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.sc_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.pw_off atIndex:1];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.rp_off atIndex:2];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.out_off atIndex:3];
        [enc setBuffer:tok_buf offset:tok_byte_off atIndex:4];
        [enc setBuffer:r->weight_arena_buf offset:r->layers[0].attn_norm_off atIndex:5];
        [enc setBuffer:r->cur_buf offset:0 atIndex:6];
        [enc setBuffer:r->normed_buf offset:0 atIndex:7];
        [enc setBytes:&ep length:sizeof(ep) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:tg256];
        norm_ready = true;
    } else {
        [enc setComputePipelineState:r->pso_embed];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.sc_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.pw_off atIndex:1];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.rp_off atIndex:2];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.out_off atIndex:3];
        [enc setBuffer:tok_buf offset:tok_byte_off atIndex:4];
        [enc setBuffer:r->cur_buf offset:0 atIndex:5];
        [enc setBytes:&ep length:sizeof(ep) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(r->padded_embd / 256u, batch_m, 1)
            threadsPerThreadgroup:tg256];
    }

    Cb4FastNormParamsC np{c.n_embd, r->padded_embd, batch_m, c.rms_norm_eps};
    Cb4FastSwigluParamsC sp{c.n_ff, r->gu_stride, r->padded_ff, batch_m};

    uint32_t conv_ch = c.ssm_inner_size + 2u * c.ssm_group_count * c.ssm_state_size;
    uint32_t dv = c.ssm_dt_rank > 0u ? (c.ssm_inner_size / c.ssm_dt_rank) : 0u;
    uint32_t conv_tgs = (conv_ch + 255u) / 256u;
    uint32_t ab_tgs = (2u * c.ssm_dt_rank + 7u) / 8u;
    uint32_t kv_dim = c.n_head_kv * c.head_dim;

    bool can_split_dnet = fuse_m1 && (c.ssm_conv_kernel > 0u) && (c.ssm_state_size <= 256u)
        && (dv >= 32u && dv <= 256u && (dv & 3u) == 0u)
        && (c.ssm_dt_rank >= c.ssm_group_count) && ((c.ssm_dt_rank % c.ssm_group_count) == 0u)
        && (c.ssm_inner_size == c.ssm_dt_rank * dv);
    bool can_fuse_gqa = fuse_m1 && (c.n_head_kv > 0u) && ((c.n_head % c.n_head_kv) == 0u)
        && (c.head_dim <= 512u);

    for (uint32_t il = 0; il < c.n_layer; ++il) {
        const auto& L = r->layers[il];
        if (!norm_ready) {
            [enc setComputePipelineState:r->pso_rms_norm];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:L.attn_norm_off atIndex:1];
            [enc setBuffer:r->normed_buf offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, batch_m, 1) threadsPerThreadgroup:tg256];
        }
        norm_ready = false;

        encode_cb4_mat(
            r, enc, L.w_in, r->normed_buf, 0, r->padded_embd,
            r->qkv_z_buf, 0, r->qkv_z_stride, batch_m, 0u
        );

        if (L.is_recurrent) {
            if (can_split_dnet) {
                Cb4FastDnetM1ParamsC dm1{
                    c.ssm_group_count, c.ssm_dt_rank, c.ssm_state_size, dv, conv_ch,
                    c.ssm_conv_kernel, conv_tgs, r->padded_embd, r->ab_half_stride,
                    r->padded_inner, L.ab_is_strided32, c.rms_norm_eps
                };
                [enc setComputePipelineState:r->pso_dnet_prep_m1];
                [enc setBuffer:r->qkv_z_buf offset:0 atIndex:0];
                [enc setBuffer:r->weight_arena_buf offset:L.vec_off[0] atIndex:1];
                [enc setBuffer:s->conv_states[il] offset:0 atIndex:2];
                [enc setBuffer:r->conv_silu_buf offset:0 atIndex:3];
                [enc setBuffer:r->weight_arena_buf offset:L.ab_off atIndex:4];
                [enc setBuffer:r->normed_buf offset:0 atIndex:5];
                [enc setBuffer:r->ab_buf offset:0 atIndex:6];
                [enc setBytes:&dm1 length:sizeof(dm1) atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(conv_tgs + ab_tgs, 1, 1)
                    threadsPerThreadgroup:tg256];

                [enc setComputePipelineState:r->pso_dnet_state_m1];
                [enc setBuffer:r->conv_silu_buf offset:0 atIndex:0];
                [enc setBuffer:r->ab_buf offset:0 atIndex:1];
                [enc setBuffer:r->weight_arena_buf offset:L.vec_off[1] atIndex:2];
                [enc setBuffer:r->weight_arena_buf offset:L.vec_off[2] atIndex:3];
                [enc setBuffer:s->ssm_states[il] offset:0 atIndex:4];
                [enc setBuffer:r->o_raw_buf offset:0 atIndex:5];
                [enc setBytes:&dm1 length:sizeof(dm1) atIndex:6];
                [enc dispatchThreadgroups:MTLSizeMake(c.ssm_dt_rank * 4u, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                [enc setComputePipelineState:r->pso_dnet_out_m1];
                [enc setBuffer:r->o_raw_buf offset:0 atIndex:0];
                [enc setBuffer:r->qkv_z_buf offset:0 atIndex:1];
                [enc setBuffer:r->weight_arena_buf offset:L.vec_off[3] atIndex:2];
                [enc setBuffer:r->y_concat_buf offset:0 atIndex:3];
                [enc setBytes:&dm1 length:sizeof(dm1) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(c.ssm_dt_rank, 1, 1)
                    threadsPerThreadgroup:tg256];
            } else {
                Cb4FastAbParamsC abp{
                    2u * c.ssm_dt_rank, r->padded_embd, r->padded_embd, r->ab_stride,
                    r->ab_half_stride, c.ssm_dt_rank, batch_m, L.ab_is_strided32
                };
                [enc setComputePipelineState:r->pso_ab_gemv];
                [enc setBuffer:r->weight_arena_buf offset:L.ab_off atIndex:0];
                [enc setBuffer:r->normed_buf offset:0 atIndex:1];
                [enc setBuffer:r->ab_buf offset:0 atIndex:2];
                [enc setBytes:&abp length:sizeof(abp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(ab_tgs, batch_m, 1)
                    threadsPerThreadgroup:tg256];

                Cb4FastDnetBatchParamsC dbp{
                    c.ssm_group_count, c.ssm_dt_rank, c.ssm_state_size, dv, conv_ch,
                    c.ssm_conv_kernel, batch_m, r->qkv_z_stride, r->conv_stride,
                    r->ab_stride, r->ab_half_stride, r->padded_inner, c.rms_norm_eps, 0, 0, 0
                };
                [enc setComputePipelineState:r->pso_dnet_conv_batch];
                [enc setBuffer:r->qkv_z_buf offset:0 atIndex:0];
                [enc setBuffer:r->weight_arena_buf offset:L.vec_off[0] atIndex:1];
                [enc setBuffer:s->conv_states[il] offset:0 atIndex:2];
                [enc setBuffer:r->conv_silu_buf offset:0 atIndex:3];
                [enc setBytes:&dbp length:sizeof(dbp) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(conv_tgs, 1, 1)
                    threadsPerThreadgroup:tg256];

                [enc setComputePipelineState:r->pso_dnet_heads_batch];
                [enc setBuffer:r->conv_silu_buf offset:0 atIndex:0];
                [enc setBuffer:r->qkv_z_buf offset:0 atIndex:1];
                [enc setBuffer:r->ab_buf offset:0 atIndex:2];
                [enc setBuffer:r->weight_arena_buf offset:L.vec_off[1] atIndex:3];
                [enc setBuffer:r->weight_arena_buf offset:L.vec_off[2] atIndex:4];
                [enc setBuffer:r->weight_arena_buf offset:L.vec_off[3] atIndex:5];
                [enc setBuffer:s->ssm_states[il] offset:0 atIndex:6];
                [enc setBuffer:r->y_concat_buf offset:0 atIndex:7];
                [enc setBytes:&dbp length:sizeof(dbp) atIndex:8];
                [enc dispatchThreadgroups:MTLSizeMake(c.ssm_dt_rank, 1, 1)
                    threadsPerThreadgroup:tg256];
            }
            encode_cb4_mat(
                r, enc, L.w_out, r->y_concat_buf, 0, r->padded_inner,
                r->delta_buf, 0, r->padded_embd, batch_m, 0u
            );
        } else {
            auto vec_buf = [&](int idx) -> id<MTLBuffer> {
                return L.has_vec[idx] ? r->weight_arena_buf : r->dummy_buf;
            };
            auto vec_off = [&](int idx) -> size_t {
                return L.has_vec[idx] ? L.vec_off[idx] : 0u;
            };
            Cb4FastGqaBatchParamsC gp{
                c.n_head, c.n_head_kv, c.head_dim, pos_base, c.rope_dim, L.gqa_flags,
                batch_m, r->qkv_z_stride, L.w_in.split0, kv_dim, r->padded_att,
                c.rope_freq_base, c.rms_norm_eps, 0, 0, 0
            };
            if (can_fuse_gqa) {
                [enc setComputePipelineState:r->pso_gqa_fused_m1];
                [enc setBuffer:r->qkv_z_buf offset:0 atIndex:0];
                [enc setBuffer:vec_buf(0) offset:vec_off(0) atIndex:1];
                [enc setBuffer:vec_buf(1) offset:vec_off(1) atIndex:2];
                [enc setBuffer:vec_buf(2) offset:vec_off(2) atIndex:3];
                [enc setBuffer:vec_buf(3) offset:vec_off(3) atIndex:4];
                [enc setBuffer:vec_buf(4) offset:vec_off(4) atIndex:5];
                [enc setBuffer:s->k_caches[il] offset:0 atIndex:6];
                [enc setBuffer:s->v_caches[il] offset:0 atIndex:7];
                [enc setBuffer:r->att_cat_buf offset:0 atIndex:8];
                [enc setBytes:&gp length:sizeof(gp) atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake(c.n_head, 1, 1)
                    threadsPerThreadgroup:tg256];
            } else {
                [enc setComputePipelineState:r->pso_gqa_prep_batch];
                [enc setBuffer:r->qkv_z_buf offset:0 atIndex:0];
                [enc setBuffer:vec_buf(0) offset:vec_off(0) atIndex:1];
                [enc setBuffer:vec_buf(1) offset:vec_off(1) atIndex:2];
                [enc setBuffer:vec_buf(2) offset:vec_off(2) atIndex:3];
                [enc setBuffer:vec_buf(3) offset:vec_off(3) atIndex:4];
                [enc setBuffer:vec_buf(4) offset:vec_off(4) atIndex:5];
                [enc setBuffer:r->q_cur_buf offset:0 atIndex:6];
                [enc setBuffer:r->q_gate_buf offset:0 atIndex:7];
                [enc setBuffer:s->k_caches[il] offset:0 atIndex:8];
                [enc setBuffer:s->v_caches[il] offset:0 atIndex:9];
                [enc setBytes:&gp length:sizeof(gp) atIndex:10];
                [enc dispatchThreadgroups:MTLSizeMake(c.n_head + c.n_head_kv, batch_m, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc setComputePipelineState:r->pso_gqa_attn_batch];
                [enc setBuffer:r->q_cur_buf offset:0 atIndex:0];
                [enc setBuffer:r->q_gate_buf offset:0 atIndex:1];
                [enc setBuffer:s->k_caches[il] offset:0 atIndex:2];
                [enc setBuffer:s->v_caches[il] offset:0 atIndex:3];
                [enc setBuffer:r->att_cat_buf offset:0 atIndex:4];
                [enc setBytes:&gp length:sizeof(gp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(c.n_head, batch_m, 1)
                    threadsPerThreadgroup:tg256];
            }
            encode_cb4_mat(
                r, enc, L.w_out, r->att_cat_buf, 0, r->padded_att,
                r->delta_buf, 0, r->padded_embd, batch_m, 0u
            );
        }

        encode_add_rms_norm(r, enc, L.post_attn_norm_off, batch_m);

        if (can_fuse_gu) {
            Cb4FastFfnGuParamsC gp{};
            gp.n_ff = c.n_ff;
            gp.padded_ff = r->padded_ff;
            gp.padded_cols = L.w_ffn_gu.padded_cols;
            std::memcpy(gp.poly_g, L.w_ffn_gu.poly[0], sizeof(gp.poly_g));
            std::memcpy(gp.poly_u, L.w_ffn_gu.poly[1], sizeof(gp.poly_u));

            uint32_t gu_thr = (r->dispatch_mode == 4u) ? 128u : 64u;
            uint32_t tiles_per_tg = gu_thr >> 5u;
            uint32_t n_tiles_g = c.n_ff >> 1u;
            [enc setComputePipelineState:r->pso_ffn_gu_m1];
            [enc setBuffer:r->weight_arena_buf offset:L.w_ffn_gu.sc_off atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:L.w_ffn_gu.pw_off atIndex:1];
            [enc setBuffer:r->weight_arena_buf offset:L.w_ffn_gu.rp_off atIndex:2];
            [enc setBuffer:r->weight_arena_buf offset:L.w_ffn_gu.out_off atIndex:3];
            [enc setBuffer:r->normed_buf offset:0 atIndex:4];
            [enc setBuffer:r->ffn_act_buf offset:0 atIndex:5];
            [enc setBytes:&gp length:sizeof(gp) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake((n_tiles_g + tiles_per_tg - 1u) / tiles_per_tg, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(gu_thr, 1, 1)];
        } else {
            encode_cb4_mat(
                r, enc, L.w_ffn_gu, r->normed_buf, 0, r->padded_embd,
                r->ffn_gu_buf, 0, r->gu_stride, batch_m, 0u
            );
            [enc setComputePipelineState:r->pso_swiglu];
            [enc setBuffer:r->ffn_gu_buf offset:0 atIndex:0];
            [enc setBuffer:r->ffn_act_buf offset:0 atIndex:1];
            [enc setBytes:&sp length:sizeof(sp) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(r->padded_ff / 256u, batch_m, 1)
                threadsPerThreadgroup:tg256];
        }

        bool has_next_norm = (il + 1u < c.n_layer) || (compute_mode != 0u);
        if (has_next_norm) {
            size_t next_norm_off = (il + 1u < c.n_layer) ? r->layers[il + 1u].attn_norm_off : r->output_norm_off;
            encode_cb4_mat(
                r, enc, L.w_ffn_dn, r->ffn_act_buf, 0, r->padded_ff,
                r->delta_buf, 0, r->padded_embd, batch_m, 0u
            );
            encode_add_rms_norm(r, enc, next_norm_off, batch_m);
            norm_ready = true;
        } else {
            encode_cb4_mat(
                r, enc, L.w_ffn_dn, r->ffn_act_buf, 0, r->padded_ff,
                r->cur_buf, 0, r->padded_embd, batch_m, 1u
            );
        }
    }

    if (compute_mode != 0u) {
        uint32_t target_m = (compute_mode == 2u) ? (batch_m - 1u) : 0u;
        size_t norm_off = 0u;
        if (!norm_ready) {
            size_t cur_off = size_t(target_m) * size_t(r->padded_embd) * sizeof(float);
            Cb4FastNormParamsC onp{c.n_embd, r->padded_embd, 1u, c.rms_norm_eps};
            [enc setComputePipelineState:r->pso_rms_norm];
            [enc setBuffer:r->cur_buf offset:cur_off atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:r->output_norm_off atIndex:1];
            [enc setBuffer:r->normed_buf offset:0 atIndex:2];
            [enc setBytes:&onp length:sizeof(onp) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:tg256];
        } else {
            norm_off = size_t(target_m) * size_t(r->padded_embd) * sizeof(float);
        }

        encode_cb4_mat(
            r, enc, r->head_mat, r->normed_buf, norm_off, r->padded_embd,
            r->logits_buf, 0, c.n_vocab, 1u, 0u
        );

        if (compute_mode == 3u) {
            uint32_t ntgs = std::min<uint32_t>(CB4_ARGMAX_TGS, std::max<uint32_t>(1u, (c.n_vocab + 255u) / 256u));
            Cb4FastArgmaxParamsC ap{c.n_vocab, slot, next_slot, ntgs};

            [enc setComputePipelineState:r->pso_argmax_s1];
            [enc setBuffer:r->logits_buf offset:0 atIndex:0];
            [enc setBuffer:r->s1_vals_buf offset:0 atIndex:1];
            [enc setBuffer:r->s1_ids_buf offset:0 atIndex:2];
            [enc setBuffer:r->s1_bad_buf offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(ntgs, 1, 1) threadsPerThreadgroup:tg256];

            [enc setComputePipelineState:r->pso_argmax_s2];
            [enc setBuffer:r->s1_vals_buf offset:0 atIndex:0];
            [enc setBuffer:r->s1_ids_buf offset:0 atIndex:1];
            [enc setBuffer:r->s1_bad_buf offset:0 atIndex:2];
            [enc setBuffer:r->out_ring_buf offset:0 atIndex:3];
            [enc setBuffer:r->err_ring_buf offset:0 atIndex:4];
            [enc setBuffer:r->tok_ring_buf offset:0 atIndex:5];
            [enc setBytes:&ap length:sizeof(ap) atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:tg256];
        }
    }
}

} // namespace

extern "C" {

int vxt_metal_cb4_fast_step_batch(
    void* raw_runner, void* raw_state, const uint32_t* tok_ids, uint32_t batch_m,
    uint32_t pos_base, uint32_t compute_mode, float* logits_out, char* err_out, size_t err_cap
) {
    @autoreleasepool {
        auto* r = static_cast<MetalCb4FastRunner*>(raw_runner);
        auto* s = static_cast<MetalCb4FastState*>(raw_state);
        if (!r || !s || !tok_ids || batch_m == 0u || batch_m > CB4_MAX_BATCH_M) {
            vxt_write_err(err_out, err_cap, "invalid arguments to vxt_metal_cb4_fast_step_batch");
            return -1;
        }
        for (uint32_t m = 0u; m < batch_m; ++m) {
            if (tok_ids[m] >= r->cfg.n_vocab) {
                vxt_write_err(err_out, err_cap, "token id out of vocabulary range");
                return -1;
            }
        }
        if (!s->ensure_kv_cap(pos_base + batch_m)) {
            vxt_write_err(err_out, err_cap, "failed to grow CB4-Fast KV cache");
            return -2;
        }
        std::memcpy([r->tok_in_buf contents], tok_ids, size_t(batch_m) * sizeof(uint32_t));

        id<MTLCommandBuffer> cmd = [r->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            vxt_write_err(err_out, err_cap, "failed to create CB4-Fast command buffer");
            return -3;
        }
        encode_cb4_fast_forward(r, s, enc, r->tok_in_buf, 0, batch_m, pos_base, compute_mode, 0u, 0xffffffffu);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            NSString* desc = [[cmd error] localizedDescription];
            vxt_write_err(err_out, err_cap, desc ? [desc UTF8String] : "CB4-Fast step_batch failed");
            return -4;
        }
        if (r->profile_enabled) {
            double gpu_ms = double([cmd GPUEndTime] - [cmd GPUStartTime]) * 1000.0;
            std::fprintf(stderr, "[CB4-PROF] step_batch m=%u mode=%u gpu_ms=%.3f us_per_tok=%.1f\n",
                batch_m, compute_mode, gpu_ms, (gpu_ms * 1000.0) / batch_m);
        }
        if (logits_out && compute_mode != 0u) {
            std::memcpy(logits_out, [r->logits_buf contents], size_t(r->cfg.n_vocab) * sizeof(float));
        }
        return 0;
    }
}

int vxt_metal_cb4_fast_checkpoint(
    void* raw_runner, void* raw_state, uint32_t restore, char* err_out, size_t err_cap
) {
    @autoreleasepool {
        auto* r = static_cast<MetalCb4FastRunner*>(raw_runner);
        auto* s = static_cast<MetalCb4FastState*>(raw_state);
        if (!r || !s) {
            vxt_write_err(err_out, err_cap, "null CB4-Fast checkpoint");
            return -1;
        }
        bool any_rec = false;
        for (uint8_t ir : s->is_rec) if (ir) { any_rec = true; break; }
        if (!any_rec) return 0;

        id<MTLCommandBuffer> cmd = [r->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (!cmd || !blit) {
            vxt_write_err(err_out, err_cap, "cannot create CB4-Fast checkpoint command");
            return -1;
        }
        for (uint32_t il = 0; il < s->n_layer; ++il) {
            if (!s->is_rec[il]) continue;
            if (s->conv_bytes[il] > 0u) {
                [blit copyFromBuffer:(restore ? s->conv_bak[il] : s->conv_states[il]) sourceOffset:0
                            toBuffer:(restore ? s->conv_states[il] : s->conv_bak[il]) destinationOffset:0
                                size:s->conv_bytes[il]];
            }
            if (s->ssm_bytes[il] > 0u) {
                [blit copyFromBuffer:(restore ? s->ssm_bak[il] : s->ssm_states[il]) sourceOffset:0
                            toBuffer:(restore ? s->ssm_states[il] : s->ssm_bak[il]) destinationOffset:0
                                size:s->ssm_bytes[il]];
            }
        }
        [blit endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            vxt_write_err(err_out, err_cap, "CB4-Fast checkpoint blit failed");
            return -1;
        }
        return 0;
    }
}

void vxt_metal_cb4_fast_ring_set_token(void* raw_runner, uint32_t slot, uint32_t tok_id) {
    auto* r = static_cast<MetalCb4FastRunner*>(raw_runner);
    if (!r) return;
    static_cast<uint32_t*>([r->tok_ring_buf contents])[slot & (CB4_RING_SLOTS - 1u)] = tok_id;
}

uint32_t vxt_metal_cb4_fast_ring_get_out(void* raw_runner, uint32_t slot) {
    auto* r = static_cast<MetalCb4FastRunner*>(raw_runner);
    if (!r) return 0u;
    return static_cast<const uint32_t*>([r->out_ring_buf contents])[slot & (CB4_RING_SLOTS - 1u)];
}

uint32_t vxt_metal_cb4_fast_ring_get_err(void* raw_runner, uint32_t slot) {
    auto* r = static_cast<MetalCb4FastRunner*>(raw_runner);
    if (!r) return 1u;
    return static_cast<const uint32_t*>([r->err_ring_buf contents])[slot & (CB4_RING_SLOTS - 1u)];
}

void* vxt_metal_cb4_fast_enqueue_async(
    void* raw_runner, void* raw_state, uint32_t base_slot, uint32_t step_count,
    uint32_t pos_base, char* err_out, size_t err_cap
) {
    @autoreleasepool {
        auto* r = static_cast<MetalCb4FastRunner*>(raw_runner);
        auto* s = static_cast<MetalCb4FastState*>(raw_state);
        if (!r || !s || step_count == 0u) {
            vxt_write_err(err_out, err_cap, "invalid arguments to vxt_metal_cb4_fast_enqueue_async");
            return nullptr;
        }
        if (s->kv_stride != 0u && pos_base + step_count + 200u >= s->cap_seq) {
            id<MTLCommandBuffer> fence = [r->queue commandBuffer];
            if (!fence) {
                vxt_write_err(err_out, err_cap, "cannot create CB4-Fast KV growth fence");
                return nullptr;
            }
            [fence commit];
            [fence waitUntilCompleted];
            if ([fence status] != MTLCommandBufferStatusCompleted) {
                vxt_write_err(err_out, err_cap, "CB4-Fast KV growth fence failed");
                return nullptr;
            }
        }
        if (!s->ensure_kv_cap(pos_base + step_count + 200u)) {
            vxt_write_err(err_out, err_cap, "failed to grow CB4-Fast KV cache in async enqueue");
            return nullptr;
        }

        id<MTLCommandBuffer> cmd = [r->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            vxt_write_err(err_out, err_cap, "failed to create async CB4-Fast command buffer");
            return nullptr;
        }
        for (uint32_t k = 0u; k < step_count; ++k) {
            uint32_t slot = (base_slot + k) & (CB4_RING_SLOTS - 1u);
            uint32_t next_slot = (base_slot + k + 1u) & (CB4_RING_SLOTS - 1u);
            size_t tok_off = size_t(slot) * sizeof(uint32_t);
            encode_cb4_fast_forward(r, s, enc, r->tok_ring_buf, tok_off, 1u, pos_base + k, 3u, slot, next_slot);
        }
        [enc endEncoding];
        if (r->profile_enabled) {
            uint32_t sc = step_count;
            [cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                double gpu_ms = double([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
                std::fprintf(stderr, "[CB4-PROF] burst=%u gpu_ms=%.3f us_per_tok=%.1f gpu_tps=%.2f\n",
                    sc, gpu_ms, (gpu_ms * 1000.0) / sc, (1000.0 * sc) / std::max(gpu_ms, 1e-6));
            }];
        }
        [cmd commit];
        return (__bridge_retained void*)cmd;
    }
}

int vxt_metal_cb4_fast_wait_cmd(void* raw_cmd, char* err_out, size_t err_cap) {
    @autoreleasepool {
        if (!raw_cmd) return 0;
        id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)raw_cmd;
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            NSString* desc = [[cmd error] localizedDescription];
            vxt_write_err(err_out, err_cap, desc ? [desc UTF8String] : "async CB4-Fast command buffer failed");
            return -1;
        }
        return 0;
    }
}

} // extern "C"
