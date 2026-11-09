#include "metal_cb4_fast.h"
#include <dispatch/dispatch.h>

namespace {

static void plan_concat_mat(const VxtCb4FastConcatMat& src, Cb4ArenaMat& dst, bool is_nr4, size_t& cursor) {
    uint32_t total_rows = 0u;
    uint32_t total_nnz = 0u;
    uint32_t pcols = src.subs[0].padded_cols;
    uint32_t cols = src.subs[0].cols;
    for (uint32_t s = 0; s < src.n_subs; ++s) {
        for (int k = 0; k < 4; ++k) dst.poly[s][k] = src.subs[s].poly[k];
        total_rows += src.subs[s].rows;
        total_nnz += src.subs[s].nnz;
    }
    for (uint32_t s = src.n_subs; s < 3u; ++s) {
        for (int k = 0; k < 4; ++k) dst.poly[s][k] = dst.poly[src.n_subs - 1u][k];
    }
    dst.rows = total_rows;
    dst.cols = cols;
    dst.padded_cols = pcols;
    dst.split0 = src.subs[0].rows;
    dst.split1 = (src.n_subs >= 2u) ? (src.subs[0].rows + src.subs[1].rows) : total_rows;
    dst.is_nr4 = is_nr4;

    size_t groups = size_t(pcols) / 128u;
    size_t steps = size_t(pcols) / 256u;
    dst.sc_off = cursor;
    if (is_nr4) {
        size_t tiles4 = (size_t(total_rows) + 3u) / 4u;
        cursor += cb4_align256(tiles4 * groups * 4u * sizeof(uint16_t));
        dst.pw_off = cursor;
        cursor += cb4_align256(tiles4 * steps * 32u * 4u * sizeof(uint32_t));
    } else {
        size_t tiles2 = (size_t(total_rows) + 1u) / 2u;
        cursor += cb4_align256(tiles2 * groups * 2u * sizeof(uint16_t));
        dst.pw_off = cursor;
        cursor += cb4_align256(tiles2 * steps * 32u * 2u * sizeof(uint32_t));
    }
    dst.rp_off = cursor;
    cursor += cb4_align256((size_t(total_rows) + 1u) * sizeof(uint32_t));
    dst.out_off = cursor;
    cursor += cb4_align256(std::max<size_t>(total_nnz, 1u) * sizeof(uint32_t));
}

static void pack_tile4_range(
    uint16_t* sc_dst, uint32_t* pw_dst,
    const uint16_t* const* row_sc, const uint32_t* const* row_pw,
    uint32_t total_rows, size_t groups, size_t words_per_row,
    size_t tile_beg, size_t tile_end
) {
    for (size_t tile = tile_beg; tile < tile_end; ++tile) {
        size_t r0 = tile * 4u;
        const uint16_t* s0 = (r0 + 0u < total_rows) ? row_sc[r0 + 0u] : nullptr;
        const uint16_t* s1 = (r0 + 1u < total_rows) ? row_sc[r0 + 1u] : nullptr;
        const uint16_t* s2 = (r0 + 2u < total_rows) ? row_sc[r0 + 2u] : nullptr;
        const uint16_t* s3 = (r0 + 3u < total_rows) ? row_sc[r0 + 3u] : nullptr;

        uint16_t* sc_t = sc_dst + tile * groups * 4u;
        if (s3) {
            for (size_t g = 0; g < groups; ++g) {
                sc_t[g * 4u + 0u] = s0[g];
                sc_t[g * 4u + 1u] = s1[g];
                sc_t[g * 4u + 2u] = s2[g];
                sc_t[g * 4u + 3u] = s3[g];
            }
        } else {
            for (size_t g = 0; g < groups; ++g) {
                sc_t[g * 4u + 0u] = s0 ? s0[g] : 0u;
                sc_t[g * 4u + 1u] = s1 ? s1[g] : 0u;
                sc_t[g * 4u + 2u] = s2 ? s2[g] : 0u;
                sc_t[g * 4u + 3u] = 0u;
            }
        }

        const uint32_t* p0 = (r0 + 0u < total_rows) ? row_pw[r0 + 0u] : nullptr;
        const uint32_t* p1 = (r0 + 1u < total_rows) ? row_pw[r0 + 1u] : nullptr;
        const uint32_t* p2 = (r0 + 2u < total_rows) ? row_pw[r0 + 2u] : nullptr;
        const uint32_t* p3 = (r0 + 3u < total_rows) ? row_pw[r0 + 3u] : nullptr;

        uint32_t* pw_t = pw_dst + tile * words_per_row * 4u;
        if (p3) {
            for (size_t w = 0; w < words_per_row; ++w) {
                pw_t[w * 4u + 0u] = p0[w];
                pw_t[w * 4u + 1u] = p1[w];
                pw_t[w * 4u + 2u] = p2[w];
                pw_t[w * 4u + 3u] = p3[w];
            }
        } else {
            for (size_t w = 0; w < words_per_row; ++w) {
                pw_t[w * 4u + 0u] = p0 ? p0[w] : 0u;
                pw_t[w * 4u + 1u] = p1 ? p1[w] : 0u;
                pw_t[w * 4u + 2u] = p2 ? p2[w] : 0u;
                pw_t[w * 4u + 3u] = 0u;
            }
        }
    }
}

static void pack_step2_mat(
    uint16_t* sc_dst, uint32_t* pw_dst,
    const uint16_t* const* row_sc, const uint32_t* const* row_pw,
    uint32_t total_rows, size_t steps
) {
    size_t tiles2 = (size_t(total_rows) + 1u) / 2u;
    for (size_t s = 0; s < steps; ++s) {
        size_t g0 = s * 2u;
        size_t w_off = s * 32u;
        for (size_t t = 0; t < tiles2; ++t) {
            size_t r0 = t * 2u;
            size_t r1 = r0 + 1u;
            const uint16_t* s0 = row_sc[r0];
            const uint16_t* s1 = (r1 < total_rows) ? row_sc[r1] : nullptr;
            const uint32_t* p0 = row_pw[r0] + w_off;
            const uint32_t* p1 = (r1 < total_rows) ? (row_pw[r1] + w_off) : nullptr;

            size_t idx = s * tiles2 + t;
            uint16_t* sc_p = sc_dst + idx * 4u;
            sc_p[0] = s0[g0];
            sc_p[1] = s1 ? s1[g0] : 0u;
            sc_p[2] = s0[g0 + 1u];
            sc_p[3] = s1 ? s1[g0 + 1u] : 0u;

            uint32_t* pw_p = pw_dst + idx * 64u;
            if (p1) {
                for (size_t lane = 0; lane < 32u; ++lane) {
                    pw_p[lane * 2u + 0u] = p0[lane];
                    pw_p[lane * 2u + 1u] = p1[lane];
                }
            } else {
                for (size_t lane = 0; lane < 32u; ++lane) {
                    pw_p[lane * 2u + 0u] = p0[lane];
                    pw_p[lane * 2u + 1u] = 0u;
                }
            }
        }
    }
}

static void pack_concat_mat(
    const VxtCb4FastConcatMat& src, const Cb4ArenaMat& dst, uint8_t* base, bool parallel_tiles
) {
    auto* sc_dst = reinterpret_cast<uint16_t*>(base + dst.sc_off);
    auto* pw_dst = reinterpret_cast<uint32_t*>(base + dst.pw_off);
    auto* rp_dst = reinterpret_cast<uint32_t*>(base + dst.rp_off);
    auto* out_dst = reinterpret_cast<uint32_t*>(base + dst.out_off);

    size_t groups = size_t(dst.padded_cols) / 128u;
    size_t steps = size_t(dst.padded_cols) / 256u;
    size_t words_per_row = steps * 32u;
    std::vector<const uint16_t*> row_sc(dst.rows);
    std::vector<const uint32_t*> row_pw(dst.rows);

    uint32_t row_cursor = 0u;
    uint32_t nnz_cursor = 0u;
    out_dst[0] = 0u;
    for (uint32_t s = 0; s < src.n_subs; ++s) {
        const auto& sm = src.subs[s];
        for (uint32_t r = 0; r < sm.rows; ++r) {
            uint32_t gr = row_cursor + r;
            row_sc[gr] = sm.scales_f16 + size_t(r) * groups;
            row_pw[gr] = sm.packed_words + size_t(r) * words_per_row;
            rp_dst[gr] = nnz_cursor + sm.outlier_row_ptr[r];
        }
        if (sm.nnz > 0u) {
            std::memcpy(out_dst + nnz_cursor, sm.outliers_packed, size_t(sm.nnz) * sizeof(uint32_t));
        }
        row_cursor += sm.rows;
        nnz_cursor += sm.nnz;
    }
    rp_dst[row_cursor] = nnz_cursor;

    const uint16_t* const* rsc_ptr = row_sc.data();
    const uint32_t* const* rpw_ptr = row_pw.data();
    uint32_t tot_rows = dst.rows;

    if (!dst.is_nr4) {
        pack_step2_mat(sc_dst, pw_dst, rsc_ptr, rpw_ptr, tot_rows, steps);
        return;
    }

    size_t tiles = (size_t(tot_rows) + 3u) / 4u;
    if (parallel_tiles && tiles >= 2048u) {
        constexpr size_t CHUNK = 1024u;
        size_t n_chunks = (tiles + CHUNK - 1u) / CHUNK;
        dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
            size_t beg = ci * CHUNK;
            size_t end = std::min<size_t>(beg + CHUNK, tiles);
            pack_tile4_range(sc_dst, pw_dst, rsc_ptr, rpw_ptr, tot_rows, groups, words_per_row, beg, end);
        });
    } else {
        pack_tile4_range(sc_dst, pw_dst, rsc_ptr, rpw_ptr, tot_rows, groups, words_per_row, 0u, tiles);
    }
}

} // namespace

extern "C" {

void* vxt_metal_cb4_fast_create(
    void* gemv_ctx_raw,
    const VxtLlmConfig* cfg_in,
    const VxtCb4FastConcatMat* embed_in,
    const float* embed_centroids,
    uint32_t has_untied_head,
    const VxtCb4FastConcatMat* head_in,
    const float* output_norm,
    const VxtCb4FastLayerUpload* layers_in,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* gemv = static_cast<MetalGemvCtx*>(gemv_ctx_raw);
        if (!gemv || !gemv->device || !gemv->queue || !gemv->lib || !cfg_in || !embed_in ||
            !embed_centroids || !output_norm || !layers_in) {
            vxt_write_err(err_out, err_cap, "invalid arguments to vxt_metal_cb4_fast_create");
            return nullptr;
        }
        id<MTLDevice> dev = gemv->device;
        id<MTLLibrary> lib = gemv->lib;

        auto* r = new MetalCb4FastRunner();
        r->device = dev;
        r->queue = gemv->queue;
        r->cfg = *cfg_in;
        if (const char* dm = std::getenv("VXG_CB4_FAST_MODE")) {
            r->dispatch_mode = uint32_t(std::atoi(dm));
        }
        if (const char* nf = std::getenv("VXG_CB4_NO_FUSE")) {
            r->disable_fusions = (std::atoi(nf) != 0);
        }
        if (const char* pf = std::getenv("VXG_PROFILE_CB4_FAST")) {
            r->profile_enabled = (std::atoi(pf) != 0);
        }

        struct PsoEntry { __strong id<MTLComputePipelineState>* dst; NSString* name; };
        PsoEntry pso_list[] = {
            {&r->pso_embed,             @"vxt_cb4_embed_exact_batch"},
            {&r->pso_embed_norm0,       @"vxt_cb4_embed_norm0_m1"},
            {&r->pso_rms_norm,          @"vxt_cb4_rms_norm_batch"},
            {&r->pso_add_rms_norm,      @"vxt_cb4_add_rms_norm_batch"},
            {&r->pso_swiglu,            @"vxt_cb4_swiglu_batch"},
            {&r->pso_ab_gemv,           @"vxt_cb4_ab_gemv_batch"},
            {&r->pso_dnet_prep_m1,      @"vxt_cb4_dnet_prep_m1"},
            {&r->pso_dnet_state_m1,     @"vxt_cb4_dnet_state_split_m1"},
            {&r->pso_dnet_out_m1,       @"vxt_cb4_dnet_out_m1"},
            {&r->pso_dnet_conv_batch,   @"vxt_cb4_dnet_conv_batch"},
            {&r->pso_dnet_heads_batch,  @"vxt_cb4_dnet_heads_batch"},
            {&r->pso_gqa_fused_m1,      @"vxt_cb4_gqa_fused_m1"},
            {&r->pso_gqa_prep_batch,    @"vxt_cb4_gqa_prep_batch"},
            {&r->pso_gqa_attn_batch,    @"vxt_cb4_gqa_attn_batch"},
            {&r->pso_gemv_nr4,          @"vxt_cb4_gemv_nr4"},
            {&r->pso_gemv_nr2,          @"vxt_cb4_gemv_nr2"},
            {&r->pso_ffn_gu_m1,         @"vxt_cb4_ffn_gu_nr2_m1"},
            {&r->pso_gemm_nr2,          @"vxt_cb4_gemm_nr2"},
            {&r->pso_argmax_s1,         @"vxt_cb4_argmax_s1"},
            {&r->pso_argmax_s2,         @"vxt_cb4_argmax_s2"},
        };
        constexpr size_t N_PSO = sizeof(pso_list) / sizeof(pso_list[0]);
        char pso_errs[N_PSO][256] = {};
        PsoEntry* pso_ptr = pso_list;
        char (*perr_ptr)[256] = pso_errs;
        dispatch_apply(N_PSO, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            @autoreleasepool {
                *(pso_ptr[i].dst) = vxt_build_pso(dev, lib, pso_ptr[i].name, perr_ptr[i], 256);
            }
        });
        for (size_t i = 0; i < N_PSO; ++i) {
            if (!*(pso_list[i].dst)) {
                vxt_write_err(err_out, err_cap, pso_errs[i][0] ? pso_errs[i] : "CB4-Fast PSO build failed");
                delete r;
                return nullptr;
            }
        }

        const VxtLlmConfig& c = r->cfg;
        uint32_t conv_ch = c.ssm_inner_size + 2u * c.ssm_group_count * c.ssm_state_size;
        r->padded_embd = cb4_pad256(c.n_embd);
        r->padded_ff = cb4_pad256(c.n_ff);
        r->padded_inner = cb4_pad256(std::max<uint32_t>(c.ssm_inner_size, 1u));
        r->padded_att = cb4_pad256(std::max<uint32_t>(c.n_head * c.head_dim, 1u));
        r->conv_stride = cb4_pad64(std::max<uint32_t>(conv_ch, 16u));
        r->gu_stride = cb4_pad64(2u * c.n_ff);
        r->ab_half_stride = cb4_pad64(std::max<uint32_t>(c.ssm_dt_rank, 1u));
        r->ab_stride = 2u * r->ab_half_stride;

        std::memcpy(r->embed_centroids, embed_centroids, sizeof(r->embed_centroids));
        r->layers.resize(c.n_layer);

        size_t arena_bytes = 0u;
        plan_concat_mat(*embed_in, r->embed_mat, true, arena_bytes);
        if (has_untied_head != 0u && head_in) {
            plan_concat_mat(*head_in, r->head_mat, true, arena_bytes);
        } else {
            r->head_mat = r->embed_mat;
        }
        r->output_norm_off = arena_bytes;
        arena_bytes += cb4_align256(size_t(c.n_embd) * sizeof(float));

        uint32_t max_in_rows = 256u;
        for (uint32_t il = 0; il < c.n_layer; ++il) {
            const auto& src = layers_in[il];
            auto& dst = r->layers[il];
            dst.is_recurrent = src.is_recurrent;
            dst.gated_q = src.gated_q;
            dst.ab_is_strided32 = src.ab_is_strided32;
            dst.attn_norm_off = arena_bytes;
            arena_bytes += cb4_align256(size_t(c.n_embd) * sizeof(float));
            dst.post_attn_norm_off = arena_bytes;
            arena_bytes += cb4_align256(size_t(c.n_embd) * sizeof(float));

            plan_concat_mat(src.w_in, dst.w_in, false, arena_bytes);
            plan_concat_mat(src.w_out, dst.w_out, false, arena_bytes);
            plan_concat_mat(src.w_ffn_gu, dst.w_ffn_gu, false, arena_bytes);
            plan_concat_mat(src.w_ffn_dn, dst.w_ffn_dn, false, arena_bytes);
            max_in_rows = std::max<uint32_t>(max_in_rows, dst.w_in.rows);

            if (src.is_recurrent && src.ab_weights_pad) {
                dst.ab_off = arena_bytes;
                arena_bytes += cb4_align256(size_t(2u * c.ssm_dt_rank) * size_t(r->padded_embd) * sizeof(float));
            }
            const float* vptrs[5] = {src.vec0, src.vec1, src.vec2, src.vec3, src.vec4};
            uint32_t vlens[5] = {src.vec0_len, src.vec1_len, src.vec2_len, src.vec3_len, src.vec4_len};
            for (int v = 0; v < 5; ++v) {
                if (vptrs[v] && vlens[v] > 0u) {
                    dst.has_vec[v] = true;
                    dst.vec_off[v] = arena_bytes;
                    arena_bytes += cb4_align256(size_t(vlens[v]) * sizeof(float));
                }
            }
            dst.gqa_flags = (src.gated_q ? 1u : 0u)
                | (c.rope_neox ? 2u : 0u)
                | (dst.has_vec[0] ? 4u : 0u)
                | (dst.has_vec[1] ? 8u : 0u)
                | (dst.has_vec[2] ? 16u : 0u)
                | (dst.has_vec[3] ? 32u : 0u)
                | (dst.has_vec[4] ? 64u : 0u);
        }
        r->qkv_z_stride = cb4_pad64(max_in_rows);

        r->weight_arena_buf = [dev newBufferWithLength:std::max<size_t>(arena_bytes, 64u)
                                               options:MTLResourceStorageModeShared];
        if (!r->weight_arena_buf) {
            vxt_write_err(err_out, err_cap, "failed to allocate CB4-Fast weight arena");
            delete r;
            return nullptr;
        }
        auto* arena = static_cast<uint8_t*>([r->weight_arena_buf contents]);

        pack_concat_mat(*embed_in, r->embed_mat, arena, true);
        if (has_untied_head != 0u && head_in) {
            pack_concat_mat(*head_in, r->head_mat, arena, true);
        }
        std::memcpy(arena + r->output_norm_off, output_norm, size_t(c.n_embd) * sizeof(float));

        const Cb4ArenaLayer* dst_layers = r->layers.data();
        uint32_t padded_embd = r->padded_embd;
        dispatch_apply(c.n_layer, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t idx) {
            const auto& src = layers_in[idx];
            const auto& dst = dst_layers[idx];
            std::memcpy(arena + dst.attn_norm_off, src.attn_norm, size_t(c.n_embd) * sizeof(float));
            std::memcpy(arena + dst.post_attn_norm_off, src.post_attn_norm, size_t(c.n_embd) * sizeof(float));
            pack_concat_mat(src.w_in, dst.w_in, arena, false);
            pack_concat_mat(src.w_out, dst.w_out, arena, false);
            pack_concat_mat(src.w_ffn_gu, dst.w_ffn_gu, arena, false);
            pack_concat_mat(src.w_ffn_dn, dst.w_ffn_dn, arena, false);
            if (src.is_recurrent && src.ab_weights_pad) {
                std::memcpy(
                    arena + dst.ab_off,
                    src.ab_weights_pad,
                    size_t(2u * c.ssm_dt_rank) * size_t(padded_embd) * sizeof(float)
                );
            }
            const float* vptrs[5] = {src.vec0, src.vec1, src.vec2, src.vec3, src.vec4};
            uint32_t vlens[5] = {src.vec0_len, src.vec1_len, src.vec2_len, src.vec3_len, src.vec4_len};
            for (int v = 0; v < 5; ++v) {
                if (dst.has_vec[v]) {
                    std::memcpy(arena + dst.vec_off[v], vptrs[v], size_t(vlens[v]) * sizeof(float));
                }
            }
        });

        size_t att_dim = std::max<size_t>(c.n_head * c.head_dim, 16u);
        r->dummy_buf = cb4_alloc_zeroed(dev, 256);
        r->tok_in_buf = cb4_alloc_zeroed(dev, CB4_MAX_BATCH_M * sizeof(uint32_t));
        r->mrope_pos_buf = cb4_alloc_zeroed(dev, CB4_MAX_BATCH_M * 4u * sizeof(uint32_t));
        r->tok_ring_buf = cb4_alloc_zeroed(dev, CB4_RING_SLOTS * sizeof(uint32_t));
        r->out_ring_buf = cb4_alloc_zeroed(dev, CB4_RING_SLOTS * sizeof(uint32_t));
        r->err_ring_buf = cb4_alloc_zeroed(dev, CB4_RING_SLOTS * sizeof(uint32_t));
        r->s1_vals_buf = cb4_alloc_zeroed(dev, CB4_ARGMAX_TGS * sizeof(float));
        r->s1_ids_buf = cb4_alloc_zeroed(dev, CB4_ARGMAX_TGS * sizeof(uint32_t));
        r->s1_bad_buf = cb4_alloc_zeroed(dev, CB4_ARGMAX_TGS * sizeof(uint32_t));

        r->cur_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->padded_embd * sizeof(float));
        r->delta_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->padded_embd * sizeof(float));
        r->normed_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->padded_embd * sizeof(float));
        r->qkv_z_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->qkv_z_stride * sizeof(float));
        r->ab_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->ab_stride * sizeof(float));
        r->conv_silu_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->conv_stride * sizeof(float));
        r->o_raw_buf = cb4_alloc_zeroed(dev, std::max<size_t>(c.ssm_inner_size, 16u) * sizeof(float));
        r->y_concat_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->padded_inner * sizeof(float));
        r->att_cat_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->padded_att * sizeof(float));
        r->q_cur_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * att_dim * sizeof(float));
        r->q_gate_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * att_dim * sizeof(float));
        r->ffn_gu_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->gu_stride * sizeof(float));
        r->ffn_act_buf = cb4_alloc_zeroed(dev, size_t(CB4_MAX_BATCH_M) * r->padded_ff * sizeof(float));
        r->logits_buf = cb4_alloc_zeroed(dev, size_t(c.n_vocab) * sizeof(float));

        if (!r->dummy_buf || !r->tok_in_buf || !r->mrope_pos_buf || !r->tok_ring_buf || !r->out_ring_buf ||
            !r->err_ring_buf || !r->s1_vals_buf || !r->s1_ids_buf || !r->s1_bad_buf ||
            !r->cur_buf || !r->delta_buf || !r->normed_buf || !r->qkv_z_buf || !r->ab_buf ||
            !r->conv_silu_buf || !r->o_raw_buf || !r->y_concat_buf || !r->att_cat_buf ||
            !r->q_cur_buf || !r->q_gate_buf || !r->ffn_gu_buf || !r->ffn_act_buf || !r->logits_buf) {
            vxt_write_err(err_out, err_cap, "failed to allocate CB4-Fast workspace buffers");
            delete r;
            return nullptr;
        }

        // Pre-map weight_arena_buf in the GPU IOMMU page tables during model creation so
        // the first prefill does not pay a ~27 ms driver page-table residency stall.
        id<MTLCommandBuffer> warm_cmd = [r->queue commandBuffer];
        id<MTLBlitCommandEncoder> warm_blit = [warm_cmd blitCommandEncoder];
        if (warm_cmd && warm_blit) {
            [warm_blit copyFromBuffer:r->weight_arena_buf sourceOffset:0
                             toBuffer:r->dummy_buf destinationOffset:0 size:64];
            [warm_blit endEncoding];
            [warm_cmd commit];
            [warm_cmd waitUntilCompleted];
        }
        return r;
    }
}

void vxt_metal_cb4_fast_destroy(void* raw_runner) {
    @autoreleasepool {
        delete static_cast<MetalCb4FastRunner*>(raw_runner);
    }
}

void* vxt_metal_cb4_fast_state_create(void* raw_runner, char* err_out, size_t err_cap) {
    @autoreleasepool {
        auto* r = static_cast<MetalCb4FastRunner*>(raw_runner);
        if (!r) {
            vxt_write_err(err_out, err_cap, "null MetalCb4FastRunner");
            return nullptr;
        }
        const VxtLlmConfig& c = r->cfg;
        auto* s = new MetalCb4FastState();
        s->device = r->device;
        s->n_layer = c.n_layer;
        s->kv_stride = c.n_head_kv * c.head_dim;
        s->cap_seq = 1024u;
        s->is_rec.resize(c.n_layer, 0);
        s->conv_bytes.resize(c.n_layer, 0);
        s->ssm_bytes.resize(c.n_layer, 0);
        s->conv_states.resize(c.n_layer, nil);
        s->conv_bak.resize(c.n_layer, nil);
        s->ssm_states.resize(c.n_layer, nil);
        s->ssm_bak.resize(c.n_layer, nil);
        s->k_caches.resize(c.n_layer, nil);
        s->v_caches.resize(c.n_layer, nil);

        uint32_t conv_ch = c.ssm_inner_size + 2u * c.ssm_group_count * c.ssm_state_size;
        uint32_t hist = c.ssm_conv_kernel > 0u ? (c.ssm_conv_kernel - 1u) : 0u;
        uint32_t dv = c.ssm_dt_rank > 0u ? (c.ssm_inner_size / c.ssm_dt_rank) : 0u;

        for (uint32_t il = 0; il < c.n_layer; ++il) {
            if (r->layers[il].is_recurrent) {
                s->is_rec[il] = 1;
                size_t cb = size_t(hist) * size_t(conv_ch) * sizeof(float);
                size_t sb = size_t(c.ssm_dt_rank) * size_t(dv) * size_t(c.ssm_state_size) * sizeof(float);
                s->conv_bytes[il] = cb;
                s->ssm_bytes[il] = sb;
                s->conv_states[il] = cb4_alloc_zeroed(s->device, cb);
                s->conv_bak[il] = cb4_alloc_zeroed(s->device, cb);
                s->ssm_states[il] = cb4_alloc_zeroed(s->device, sb);
                s->ssm_bak[il] = cb4_alloc_zeroed(s->device, sb);
                if (!s->conv_states[il] || !s->conv_bak[il] || !s->ssm_states[il] || !s->ssm_bak[il]) {
                    vxt_write_err(err_out, err_cap, "failed to allocate CB4-Fast FP32 recurrent state");
                    delete s;
                    return nullptr;
                }
            } else {
                size_t kvb = size_t(s->cap_seq) * size_t(s->kv_stride) * sizeof(float);
                s->k_caches[il] = cb4_alloc_zeroed(s->device, kvb);
                s->v_caches[il] = cb4_alloc_zeroed(s->device, kvb);
                if (!s->k_caches[il] || !s->v_caches[il]) {
                    vxt_write_err(err_out, err_cap, "failed to allocate CB4-Fast FP32 KV cache");
                    delete s;
                    return nullptr;
                }
            }
        }
        return s;
    }
}

void vxt_metal_cb4_fast_state_destroy(void* raw_state) {
    @autoreleasepool {
        delete static_cast<MetalCb4FastState*>(raw_state);
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

} // extern "C"
