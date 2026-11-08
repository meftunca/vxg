#include "metal_internal.h"

#include <cstdlib>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t MAX_BATCH_M = 4u;
constexpr uint32_t RING_SLOTS = 256u;
constexpr uint32_t STAGE2_TGS = 512u;

struct TurboLlmMatParamsC {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t cols;
    uint32_t flags;
    uint32_t norm_dim;
    float norm_eps;
    uint32_t next_padded_cols;
    uint32_t sync_slot;
    uint32_t num_tgs;
    uint32_t _pad;
};

struct TurboEmbedParamsC {
    uint32_t cols;
    uint32_t padded_cols;
    uint32_t bits;
    uint32_t batch_m;
};

struct TurboEmbedNormParamsC {
    uint32_t cols;
    uint32_t padded_cols;
    uint32_t bits;
    float eps;
};

struct TurboNormParamsC {
    uint32_t dim;
    uint32_t padded_cols;
    float eps;
    uint32_t batch_m;
};

struct TurboTailNormParamsC {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t next_padded_cols;
    uint32_t flags;
    uint32_t tile_sync_base;
    uint32_t mat_sync_slot;
    float eps;
    uint32_t _pad;
};

struct TurboSwigluParamsC {
    uint32_t n_ff;
    uint32_t padded_cols;
    uint32_t gu_stride;
    uint32_t batch_m;
};

struct TurboRaw16ParamsC {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t batch_m;
    uint32_t _pad;
};

struct TurboGemmParamsC {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t batch_m;
    uint32_t tiles_per_sg;
};

struct TurboDeltaConvParamsC {
    uint32_t conv_ch;
    uint32_t k_sz;
    uint32_t qkv_gate_stride;
    uint32_t batch_m;
    uint32_t save_backup;
};

struct TurboDeltaHeadsParamsC {
    uint32_t n_kh;
    uint32_t n_vh;
    uint32_t dk;
    uint32_t dv;
    uint32_t qkv_stride;
    uint32_t z_stride;
    uint32_t padded_inner;
    uint32_t padded_embd;
    uint32_t batch_m;
    uint32_t save_backup;
    float eps;
    uint32_t fuse_out;
    uint32_t sync_slot;
};

struct TurboGqaParamsC {
    uint32_t nh;
    uint32_t nkv;
    uint32_t hd;
    uint32_t pos_base;
    uint32_t rope_dim;
    uint32_t flags;
    uint32_t qkv_stride;
    uint32_t kv_stride;
    uint32_t q_raw_dim;
    uint32_t padded_qdim;
    uint32_t batch_m;
    float rope_freq_base;
    float eps;
};

struct TurboArgmaxParamsC {
    uint32_t n_vocab;
    uint32_t n_blocks;
    uint32_t batch_m;
    uint32_t write_next_slot;
    float presence_pen;
    uint32_t d1;
    uint32_t d2;
    uint32_t d3;
};

struct TurboSketchBuildParamsC {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t bits;
    uint32_t stride_tiles8;
};

struct TurboSketchEvalParamsC {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t n_blocks;
    uint32_t stride_tiles8;
    float presence_pen;
    uint32_t d1;
    uint32_t d2;
    uint32_t d3;
};

struct TurboSketchStage2ParamsC {
    uint32_t rows;
    uint32_t padded_cols;
    uint32_t bits;
    float presence_pen;
    uint32_t stage2_tgs;
    uint32_t d1;
    uint32_t d2;
    uint32_t d3;
};

struct ArenaMat {
    size_t scales_off = 0;
    size_t packed_off = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t padded_cols = 0;
    uint32_t bits = 0;
};

struct ArenaLayerDesc {
    uint32_t is_recurrent = 0;
    uint32_t gated_q = 0;
    uint32_t gqa_flags = 0;
    uint32_t has_split_gate = 0;
    size_t attn_norm_off = 0;
    size_t post_attn_norm_off = 0;
    ArenaMat w_qkv{};
    ArenaMat w_gate_sep{};
    size_t ab_rot_f16_off = 0;
    ArenaMat w_out{};
    size_t vec0_off = 0;
    size_t vec1_off = 0;
    size_t vec2_off = 0;
    size_t vec3_off = 0;
    size_t vec4_off = 0;
    ArenaMat w_ffn_gu{};
    ArenaMat w_ffn_down{};
};

id<MTLBuffer> alloc_shared_bytes(id<MTLDevice> dev, size_t bytes) {
    size_t b = std::max<size_t>((bytes + 63u) & ~size_t(63u), 64u);
    id<MTLBuffer> buf = [dev newBufferWithLength:b options:MTLResourceStorageModeShared];
    if (buf) {
        std::memset([buf contents], 0, b);
    }
    return buf;
}

struct MetalTurboRunner {
    MetalGemvCtx gemv;
    VxtLlmConfig cfg{};
    bool has_coherent_dev = false;
    bool can_dnet_fused16 = false;
    bool can_t25_sk2_512 = false;
    bool can_t3_sk2_512 = false;
    bool can_tg512_gu = false;
    uint32_t n_tg_s1 = 0;
    uint32_t stride_tiles8 = 0;
    uint32_t tail_sync_base = 0;

    id<MTLBuffer> weight_arena_buf = nil;
    ArenaMat embed_mat{};
    ArenaMat output_head_mat{};
    size_t output_norm_off = 0;
    std::vector<ArenaLayerDesc> layers;

    id<MTLComputePipelineState> pso_build_lut_bank = nil;
    id<MTLComputePipelineState> pso_embed_turbo = nil;
    id<MTLComputePipelineState> pso_embed_norm0_m1 = nil;
    id<MTLComputePipelineState> pso_rms_norm_turbo = nil;
    id<MTLComputePipelineState> pso_rms_norm_wait_m1 = nil;
    id<MTLComputePipelineState> pso_add_rms_norm_turbo = nil;
    id<MTLComputePipelineState> pso_swiglu_turbo = nil;
    id<MTLComputePipelineState> pso_llm_raw16_rot = nil;
    id<MTLComputePipelineState> pso_m1_t2_1sg = nil;
    id<MTLComputePipelineState> pso_m1_t2_sk2 = nil;
    id<MTLComputePipelineState> pso_m1_t2_sk4 = nil;
    id<MTLComputePipelineState> pso_m1_t25_1sg = nil;
    id<MTLComputePipelineState> pso_m1_t25_sk2 = nil;
    id<MTLComputePipelineState> pso_m1_t25_sk4 = nil;
    id<MTLComputePipelineState> pso_m1_t3_1sg = nil;
    id<MTLComputePipelineState> pso_m1_t3_sk2 = nil;
    id<MTLComputePipelineState> pso_m1_t3_sk4 = nil;
    id<MTLComputePipelineState> pso_m1_ffn_gu_t2 = nil;
    id<MTLComputePipelineState> pso_m1_ffn_gu_t25 = nil;
    id<MTLComputePipelineState> pso_m1_ffn_gu_t3 = nil;
    id<MTLComputePipelineState> pso_dnet_m1_prep16 = nil;
    id<MTLComputePipelineState> pso_dnet_m1_state64 = nil;
    id<MTLComputePipelineState> pso_dnet_m1_out2 = nil;
    id<MTLComputePipelineState> pso_dnet_m1_fused16 = nil;
    id<MTLComputePipelineState> pso_dnet_m1_fused64 = nil;
    id<MTLComputePipelineState> pso_llm_turbo2_gemm4 = nil;
    id<MTLComputePipelineState> pso_llm_turbo25_gemm4 = nil;
    id<MTLComputePipelineState> pso_llm_turbo3_gemm4 = nil;
    id<MTLComputePipelineState> pso_deltanet_conv_batch = nil;
    id<MTLComputePipelineState> pso_deltanet_fused_1to1 = nil;
    id<MTLComputePipelineState> pso_deltanet_heads_fp16 = nil;
    id<MTLComputePipelineState> pso_gqa_prep_fp16 = nil;
    id<MTLComputePipelineState> pso_gqa_attn_fp16 = nil;
    id<MTLComputePipelineState> pso_gqa_fused_m1 = nil;
    id<MTLComputePipelineState> pso_argmax_s1 = nil;
    id<MTLComputePipelineState> pso_argmax_s2 = nil;
    id<MTLComputePipelineState> pso_build_sketch = nil;
    id<MTLComputePipelineState> pso_quant_x_sketch = nil;
    id<MTLComputePipelineState> pso_sketch_s1 = nil;
    id<MTLComputePipelineState> pso_sketch_thresh = nil;
    id<MTLComputePipelineState> pso_sketch_s2_t2 = nil;
    id<MTLComputePipelineState> pso_sketch_s2_t25 = nil;
    id<MTLComputePipelineState> pso_sketch_s2_t3 = nil;
    id<MTLComputePipelineState> pso_final_norm_quant = nil;
    id<MTLComputePipelineState> pso_final_norm_quant_wait_m1 = nil;

    id<MTLBuffer> lut_bank_buf = nil;
    id<MTLBuffer> caps_buf = nil;
    id<MTLBuffer> cur_buf = nil;
    id<MTLBuffer> x_rot_buf = nil;
    id<MTLBuffer> x_rot_tmp_buf = nil;
    id<MTLBuffer> x_rot_h_buf = nil;
    id<MTLBuffer> x_rot_tmp_h_buf = nil;
    id<MTLBuffer> attn_out_buf = nil;
    id<MTLBuffer> ffn_gu_buf = nil;
    id<MTLBuffer> ffn_out_buf = nil;
    id<MTLBuffer> sync_cnt_buf = nil;
    id<MTLBuffer> conv_silu_buf = nil;
    id<MTLBuffer> qkv_gqa_buf = nil;
    id<MTLBuffer> kv_gqa_buf = nil;
    id<MTLBuffer> q_cur_buf = nil;
    id<MTLBuffer> q_gate_buf = nil;
    id<MTLBuffer> logits_buf = nil;

    id<MTLBuffer> sketch_signs_buf = nil;
    id<MTLBuffer> sketch_l1_buf = nil;
    id<MTLBuffer> x_planes_buf = nil;
    id<MTLBuffer> x_qscale_buf = nil;
    id<MTLBuffer> blk_sketch_max_buf = nil;
    id<MTLBuffer> surv_blk_buf = nil;
    id<MTLBuffer> blk_vals_buf = nil;
    id<MTLBuffer> blk_idxs_buf = nil;

    id<MTLBuffer> tok_in_buf = nil;
    id<MTLBuffer> out_ids_buf = nil;
    id<MTLBuffer> tok_ring_buf = nil;
    id<MTLBuffer> out_ring_buf = nil;
};

struct MetalTurboState {
    id<MTLDevice> device = nil;
    uint32_t n_layer = 0;
    uint32_t n_vocab = 0;
    uint32_t kv_stride = 0;
    uint32_t cap_seq = 0;
    float presence_pen = 0.0f;
    id<MTLBuffer> seen_vocab_buf = nil;
    std::vector<uint8_t> is_rec;
    std::vector<size_t> conv_bytes;
    std::vector<size_t> ssm_bytes;

    std::vector<id<MTLBuffer>> conv_states;
    std::vector<id<MTLBuffer>> conv_bak;
    std::vector<id<MTLBuffer>> ssm_states;
    std::vector<id<MTLBuffer>> ssm_bak;
    std::vector<id<MTLBuffer>> qkv_hist;
    std::vector<id<MTLBuffer>> z_sep_hist;
    std::vector<id<MTLBuffer>> ab_hist;
    std::vector<id<MTLBuffer>> k_caches;
    std::vector<id<MTLBuffer>> v_caches;

    bool ensure_kv_cap(uint32_t max_pos) {
        if (max_pos < cap_seq || kv_stride == 0) return true;
        uint32_t new_cap = std::max<uint32_t>(cap_seq * 2u, max_pos + 512u);
        size_t old_bytes = size_t(cap_seq) * size_t(kv_stride) * sizeof(uint16_t);
        size_t new_bytes = size_t(new_cap) * size_t(kv_stride) * sizeof(uint16_t);
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (is_rec[il]) continue;
            id<MTLBuffer> nk = alloc_shared_bytes(device, new_bytes);
            id<MTLBuffer> nv = alloc_shared_bytes(device, new_bytes);
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

uint32_t encode_turbo_mat(
    MetalTurboRunner* r,
    id<MTLComputeCommandEncoder> enc,
    const ArenaMat& m,
    id<MTLBuffer> x_rot_f_in,
    id<MTLBuffer> x_rot_h_in,
    size_t x_elem_off,
    id<MTLBuffer> y_buf,
    size_t y_byte_off,
    uint32_t batch_m,
    bool add_residual = false,
    uint32_t epi_mode = 0u,
    uint32_t sync_slot = 0u,
    size_t norm_w_off = 0u,
    uint32_t next_padded_cols = 0u,
    bool epinorm_write_f = false
) {
    uint32_t ec_flag = (m.bits == 4u) ? 256u : 0u;
    uint32_t res_flag = add_residual ? 512u : 0u;
    if (batch_m == 1u) {
        uint32_t steps = m.padded_cols >> 8u;
        uint32_t total_tiles = (m.rows + 3u) / 4u;
        bool use_t25_512 = (m.bits == 5u && r->can_t25_sk2_512 && steps >= 2u && m.rows >= 4096u && m.rows <= 16384u);
        bool use_t3_512 = ((m.bits == 3u || m.bits == 4u) && r->can_t3_sk2_512 && steps >= 2u && m.rows >= 4096u && m.rows <= 16384u);
        bool use_sk2 = (!use_t25_512 && !use_t3_512 && steps >= 2u && m.rows <= 16384u);
        uint32_t tg_threads = 256u;
        uint32_t groups = 1u;
        id<MTLComputePipelineState> pso = nil;
        if (m.bits == 2u) {
            if (use_sk2) { tg_threads = 128u; groups = (total_tiles + 1u) / 2u; pso = r->pso_m1_t2_sk2; }
            else { tg_threads = 256u; groups = (total_tiles + 7u) / 8u; pso = r->pso_m1_t2_1sg; }
        } else if (m.bits == 5u) {
            if (use_t25_512) { tg_threads = 512u; groups = (total_tiles + 7u) / 8u; pso = r->pso_m1_t25_sk4; }
            else if (use_sk2) { tg_threads = 256u; groups = (total_tiles + 3u) / 4u; pso = r->pso_m1_t25_sk2; }
            else { tg_threads = 256u; groups = (total_tiles + 7u) / 8u; pso = r->pso_m1_t25_1sg; }
        } else {
            if (use_t3_512) { tg_threads = 512u; groups = (total_tiles + 7u) / 8u; pso = r->pso_m1_t3_sk4; }
            else if (use_sk2) { tg_threads = 256u; groups = (total_tiles + 3u) / 4u; pso = r->pso_m1_t3_sk2; }
            else { tg_threads = 256u; groups = (total_tiles + 7u) / 8u; pso = r->pso_m1_t3_1sg; }
        }
        bool has_epi_bufs = (use_sk2 || use_t25_512 || use_t3_512);
        uint32_t act_epi = has_epi_bufs ? epi_mode : 0u;
        uint32_t epi_flags = (act_epi == 1u)
            ? (1024u | (epinorm_write_f ? 2048u : 0u))
            : ((act_epi == 2u) ? 4096u : 0u);
        uint32_t mat_cols = (act_epi == 2u) ? r->cfg.n_ff : m.cols;
        TurboLlmMatParamsC p{
            m.rows, m.padded_cols, mat_cols, ec_flag | res_flag | epi_flags,
            r->cfg.n_embd, r->cfg.rms_norm_eps, (act_epi != 0u) ? next_padded_cols : m.padded_cols, sync_slot, groups, 0u
        };
        [enc setComputePipelineState:pso];
        [enc setBuffer:r->weight_arena_buf offset:m.scales_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:m.packed_off atIndex:1];
        [enc setBuffer:x_rot_h_in offset:(x_elem_off * sizeof(uint16_t)) atIndex:2];
        [enc setBuffer:y_buf offset:y_byte_off atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc setBuffer:r->lut_bank_buf offset:0 atIndex:5];
        [enc setBuffer:r->sync_cnt_buf offset:0 atIndex:6];
        if (has_epi_bufs) {
            id<MTLBuffer> h_out_buf = (act_epi == 2u) ? r->x_rot_tmp_h_buf : r->x_rot_h_buf;
            [enc setBuffer:r->weight_arena_buf offset:norm_w_off atIndex:7];
            [enc setBuffer:h_out_buf offset:0 atIndex:8];
            [enc setBuffer:r->x_rot_buf offset:0 atIndex:9];
        }
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
        return groups;
    } else {
        uint32_t steps = m.padded_cols >> 8u;
        bool use_sk2 = (steps >= 2u && m.rows <= 16384u);
        uint32_t total_tiles = (m.rows + 3u) / 4u;
        uint32_t groups = 1u;
        uint32_t tps_flags = ec_flag;
        if (use_sk2) {
            groups = (total_tiles + 3u) / 4u;
            tps_flags |= 1u | 512u;
        } else {
            uint32_t tps = (m.rows >= 65536u) ? 2u : 1u;
            uint32_t sgs_needed = (total_tiles + tps - 1u) / tps;
            groups = (sgs_needed + 7u) / 8u;
            tps_flags |= tps;
        }
        TurboGemmParamsC p{ m.rows, m.padded_cols, batch_m, tps_flags };
        id<MTLComputePipelineState> pso = (m.bits == 2u)
            ? r->pso_llm_turbo2_gemm4
            : ((m.bits == 5u) ? r->pso_llm_turbo25_gemm4 : r->pso_llm_turbo3_gemm4);
        [enc setComputePipelineState:pso];
        [enc setBuffer:r->weight_arena_buf offset:m.scales_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:m.packed_off atIndex:1];
        [enc setBuffer:x_rot_h_in offset:(x_elem_off * sizeof(uint16_t)) atIndex:2];
        [enc setBuffer:y_buf offset:y_byte_off atIndex:3];
        [enc setBytes:&p length:sizeof(p) atIndex:4];
        [enc setBuffer:r->lut_bank_buf offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        return groups;
    }
}

void encode_turbo_forward(
    MetalTurboRunner* r,
    MetalTurboState* s,
    __strong id<MTLComputeCommandEncoder>& enc,
    id<MTLBuffer> tok_buf,
    size_t tok_byte_off,
    uint32_t batch_m,
    uint32_t pos_base,
    uint32_t save_backup,
    uint32_t compute_mode,
    id<MTLBuffer> out_ids_buf,
    size_t out_ids_byte_off,
    uint32_t write_next_slot,
    uint32_t attn_mask = 0xffffffffu,
    uint32_t ffn_mask = 0xffffffffu,
    bool use_bak_state = false,
    int force_dnet_fuse = -1,
    bool is_concurrent_enc = false
) {
    const VxtLlmConfig& c = r->cfg;
    uint32_t dim = c.n_embd;
    uint32_t padded_embd = ((dim + 255u) / 256u) * 256u;
    uint32_t padded_ff = ((c.n_ff + 255u) / 256u) * 256u;
    uint32_t q_dim = c.n_head * c.head_dim;
    uint32_t padded_qdim = ((q_dim + 255u) / 256u) * 256u;
    uint32_t padded_inner = ((c.ssm_inner_size + 255u) / 256u) * 256u;
    bool fuse_m1_res = (batch_m == 1u);
    bool fuse_final_quant = (batch_m == 1u && compute_mode == 2u && r->output_head_mat.padded_cols <= 8192u);
    auto buf_bar = [&]() {
        if (is_concurrent_enc) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
    };

    // 1+2. All-GPU Embedding Decode + Layer-0 RMSNorm + Sign-Flip + FWHT32
    if (batch_m == 1u && r->embed_mat.padded_cols == padded_embd && padded_embd <= 4096u) {
        TurboEmbedNormParamsC enp{ dim, padded_embd, r->embed_mat.bits, c.rms_norm_eps };
        [enc setComputePipelineState:r->pso_embed_norm0_m1];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.scales_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.packed_off atIndex:1];
        [enc setBuffer:tok_buf offset:tok_byte_off atIndex:2];
        [enc setBuffer:r->cur_buf offset:0 atIndex:3];
        [enc setBuffer:r->weight_arena_buf offset:r->layers[0].attn_norm_off atIndex:4];
        [enc setBuffer:r->x_rot_buf offset:0 atIndex:5];
        [enc setBytes:&enp length:sizeof(enp) atIndex:6];
        [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        buf_bar();
    } else {
        TurboEmbedParamsC ep{ dim, r->embed_mat.padded_cols, r->embed_mat.bits, batch_m };
        uint32_t steps = r->embed_mat.padded_cols >> 8u;
        [enc setComputePipelineState:r->pso_embed_turbo];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.scales_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:r->embed_mat.packed_off atIndex:1];
        [enc setBuffer:tok_buf offset:tok_byte_off atIndex:2];
        [enc setBuffer:r->cur_buf offset:0 atIndex:3];
        [enc setBytes:&ep length:sizeof(ep) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(steps, batch_m, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        buf_bar();

        TurboNormParamsC np{ dim, padded_embd, c.rms_norm_eps, batch_m };
        [enc setComputePipelineState:r->pso_rms_norm_turbo];
        [enc setBuffer:r->cur_buf offset:0 atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:r->layers[0].attn_norm_off atIndex:1];
        [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
        [enc setBytes:&np length:sizeof(np) atIndex:3];
        [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, batch_m, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        buf_bar();
    }

    // 3. Transformer / Hybrid DeltaNet + FullGqa Layers
    bool next_attn_norm_ready = false;
    bool final_quant_ready = false;
    for (uint32_t il = 0; il < c.n_layer; ++il) {
        const ArenaLayerDesc& L = r->layers[il];
        bool run_attn = !fuse_m1_res || (((attn_mask >> il) & 1u) != 0u);
        bool run_ffn  = !fuse_m1_res || (((ffn_mask >> il) & 1u) != 0u);
        uint32_t dnet_sync_slot = il * c.ssm_dt_rank;
        uint32_t slot_post = r->tail_sync_base + 3u * il;
        uint32_t slot_gu   = r->tail_sync_base + 3u * il + 1u;
        uint32_t slot_next = r->tail_sync_base + 3u * il + 2u;
        bool can_epi_layer = (fuse_m1_res && r->has_coherent_dev && !use_bak_state);
        bool can_fused_gu = false;
        bool post_norm_ready = false;

        if (run_attn) {
            if (il > 0u && fuse_m1_res && !next_attn_norm_ready) {
                uint32_t norm_bm = (L.is_recurrent != 0u) ? 1u : (1u | 256u);
                TurboNormParamsC np{ dim, padded_embd, c.rms_norm_eps, norm_bm };
                [enc setComputePipelineState:r->pso_rms_norm_turbo];
                [enc setBuffer:r->cur_buf offset:0 atIndex:0];
                [enc setBuffer:r->weight_arena_buf offset:L.attn_norm_off atIndex:1];
                [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
                [enc setBytes:&np length:sizeof(np) atIndex:3];
                [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                buf_bar();
            }
            next_attn_norm_ready = false;

            if (L.is_recurrent != 0u) {
                uint32_t n_kh = c.ssm_group_count;
                uint32_t n_vh = c.ssm_dt_rank;
                uint32_t dk = c.ssm_state_size;
                uint32_t dv = c.ssm_inner_size / n_vh;
                uint32_t conv_ch = 2u * n_kh * dk + c.ssm_inner_size;
                id<MTLBuffer> c_st = use_bak_state ? s->conv_bak[il] : s->conv_states[il];
                id<MTLBuffer> s_st = use_bak_state ? s->ssm_bak[il] : s->ssm_states[il];

                encode_turbo_mat(r, enc, L.w_qkv, r->x_rot_buf, r->x_rot_h_buf, 0, s->qkv_hist[il], 0, batch_m);
                if (L.has_split_gate != 0u) {
                    encode_turbo_mat(r, enc, L.w_gate_sep, r->x_rot_buf, r->x_rot_h_buf, 0, s->z_sep_hist[il], 0, batch_m);
                }
                buf_bar();

                id<MTLBuffer> z_buf = (L.has_split_gate != 0u) ? s->z_sep_hist[il] : s->qkv_hist[il];
                size_t z_off = (L.has_split_gate != 0u) ? 0u : size_t(conv_ch) * sizeof(float);
                uint32_t z_stride = (L.has_split_gate != 0u) ? c.ssm_inner_size : L.w_qkv.rows;
                bool can_m1_dnet = (batch_m == 1u && save_backup == 0u && n_kh == n_vh && c.ssm_conv_kernel == 4u && dk == 128u && dv == 128u);
                bool use_dnet_fused64 = can_m1_dnet && r->has_coherent_dev && (force_dnet_fuse == 3);
                bool use_dnet_fused16 = !use_dnet_fused64 && can_m1_dnet && r->can_dnet_fused16 && (force_dnet_fuse < 0 || force_dnet_fuse == 2);
                uint32_t fuse_dnet_out = (!use_dnet_fused64 && !use_dnet_fused16 && force_dnet_fuse == 1 && can_m1_dnet) ? 1u : 0u;
                TurboDeltaHeadsParamsC hp{
                    n_kh, n_vh, dk, dv, L.w_qkv.rows, z_stride, padded_inner, padded_embd,
                    batch_m, save_backup, c.rms_norm_eps, fuse_dnet_out, dnet_sync_slot
                };

                if (use_dnet_fused64) {
                    [enc setComputePipelineState:r->pso_dnet_m1_fused64];
                    [enc setBuffer:s->qkv_hist[il] offset:0 atIndex:0];
                    [enc setBuffer:z_buf offset:z_off atIndex:1];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec0_off atIndex:2];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec1_off atIndex:3];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec2_off atIndex:4];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec3_off atIndex:5];
                    [enc setBuffer:c_st offset:0 atIndex:6];
                    [enc setBuffer:s_st offset:0 atIndex:7];
                    [enc setBuffer:r->weight_arena_buf offset:L.ab_rot_f16_off atIndex:8];
                    [enc setBuffer:r->x_rot_buf offset:0 atIndex:9];
                    [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:10];
                    [enc setBytes:&hp length:sizeof(hp) atIndex:11];
                    [enc setBuffer:r->conv_silu_buf offset:0 atIndex:12];
                    [enc setBuffer:r->sync_cnt_buf offset:0 atIndex:13];
                    [enc dispatchThreadgroups:MTLSizeMake(n_vh * 4u, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                } else if (use_dnet_fused16) {
                    [enc setComputePipelineState:r->pso_dnet_m1_fused16];
                    [enc setBuffer:s->qkv_hist[il] offset:0 atIndex:0];
                    [enc setBuffer:z_buf offset:z_off atIndex:1];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec0_off atIndex:2];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec1_off atIndex:3];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec2_off atIndex:4];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec3_off atIndex:5];
                    [enc setBuffer:c_st offset:0 atIndex:6];
                    [enc setBuffer:s_st offset:0 atIndex:7];
                    [enc setBuffer:r->weight_arena_buf offset:L.ab_rot_f16_off atIndex:8];
                    [enc setBuffer:r->x_rot_buf offset:0 atIndex:9];
                    [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:10];
                    [enc setBytes:&hp length:sizeof(hp) atIndex:11];
                    [enc dispatchThreadgroups:MTLSizeMake(n_vh, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
                } else if (can_m1_dnet) {
                    [enc setComputePipelineState:r->pso_dnet_m1_prep16];
                    [enc setBuffer:s->qkv_hist[il] offset:0 atIndex:0];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec0_off atIndex:1];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec1_off atIndex:2];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec2_off atIndex:3];
                    [enc setBuffer:c_st offset:0 atIndex:4];
                    [enc setBuffer:r->weight_arena_buf offset:L.ab_rot_f16_off atIndex:5];
                    [enc setBuffer:r->x_rot_buf offset:0 atIndex:6];
                    [enc setBuffer:r->conv_silu_buf offset:0 atIndex:7];
                    [enc setBytes:&hp length:sizeof(hp) atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake(n_vh, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    buf_bar();

                    [enc setComputePipelineState:r->pso_dnet_m1_state64];
                    [enc setBuffer:r->conv_silu_buf offset:0 atIndex:0];
                    [enc setBuffer:s_st offset:0 atIndex:1];
                    [enc setBytes:&hp length:sizeof(hp) atIndex:2];
                    [enc setBuffer:z_buf offset:z_off atIndex:3];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec3_off atIndex:4];
                    [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:5];
                    [enc setBuffer:r->x_rot_tmp_buf offset:0 atIndex:6];
                    [enc setBuffer:r->sync_cnt_buf offset:0 atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake(n_vh * 4u, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    if (fuse_dnet_out == 0u) {
                        buf_bar();
                        [enc setComputePipelineState:r->pso_dnet_m1_out2];
                        [enc setBuffer:r->conv_silu_buf offset:0 atIndex:0];
                        [enc setBuffer:z_buf offset:z_off atIndex:1];
                        [enc setBuffer:r->weight_arena_buf offset:L.vec3_off atIndex:2];
                        [enc setBuffer:r->x_rot_tmp_buf offset:0 atIndex:3];
                        [enc setBytes:&hp length:sizeof(hp) atIndex:4];
                        [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake((n_vh + 7u) / 8u, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    }
                } else if (n_kh == n_vh && c.ssm_conv_kernel == 4u) {
                    [enc setComputePipelineState:r->pso_deltanet_fused_1to1];
                    [enc setBuffer:s->qkv_hist[il] offset:0 atIndex:0];
                    [enc setBuffer:z_buf offset:z_off atIndex:1];
                    [enc setBuffer:s->ab_hist[il] offset:0 atIndex:2];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec0_off atIndex:3];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec1_off atIndex:4];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec2_off atIndex:5];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec3_off atIndex:6];
                    [enc setBuffer:c_st offset:0 atIndex:7];
                    [enc setBuffer:s_st offset:0 atIndex:8];
                    [enc setBuffer:r->x_rot_tmp_buf offset:0 atIndex:9];
                    [enc setBuffer:s->conv_bak[il] offset:0 atIndex:10];
                    [enc setBuffer:s->ssm_bak[il] offset:0 atIndex:11];
                    [enc setBytes:&hp length:sizeof(hp) atIndex:12];
                    [enc setBuffer:r->weight_arena_buf offset:L.ab_rot_f16_off atIndex:13];
                    [enc setBuffer:r->x_rot_buf offset:0 atIndex:14];
                    [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:15];
                    [enc dispatchThreadgroups:MTLSizeMake(n_vh, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    uint32_t ab_rows = 2u * n_vh;
                    TurboRaw16ParamsC rp{ ab_rows, padded_embd, batch_m, 0u };
                    [enc setComputePipelineState:r->pso_llm_raw16_rot];
                    [enc setBuffer:r->weight_arena_buf offset:L.ab_rot_f16_off atIndex:0];
                    [enc setBuffer:r->x_rot_buf offset:0 atIndex:1];
                    [enc setBuffer:s->ab_hist[il] offset:0 atIndex:2];
                    [enc setBytes:&rp length:sizeof(rp) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((ab_rows + 1u) / 2u, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                    TurboDeltaConvParamsC cp{ conv_ch, c.ssm_conv_kernel, L.w_qkv.rows, batch_m, save_backup };
                    [enc setComputePipelineState:r->pso_deltanet_conv_batch];
                    [enc setBuffer:s->qkv_hist[il] offset:0 atIndex:0];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec0_off atIndex:1];
                    [enc setBuffer:c_st offset:0 atIndex:2];
                    [enc setBuffer:s->conv_bak[il] offset:0 atIndex:3];
                    [enc setBuffer:r->conv_silu_buf offset:0 atIndex:4];
                    [enc setBytes:&cp length:sizeof(cp) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((conv_ch + 255u) / 256u, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    buf_bar();

                    [enc setComputePipelineState:r->pso_deltanet_heads_fp16];
                    [enc setBuffer:r->conv_silu_buf offset:0 atIndex:0];
                    [enc setBuffer:z_buf offset:z_off atIndex:1];
                    [enc setBuffer:s->ab_hist[il] offset:0 atIndex:2];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec1_off atIndex:3];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec2_off atIndex:4];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec3_off atIndex:5];
                    [enc setBuffer:s_st offset:0 atIndex:6];
                    [enc setBuffer:s->ssm_bak[il] offset:0 atIndex:7];
                    [enc setBuffer:r->x_rot_tmp_buf offset:0 atIndex:8];
                    [enc setBytes:&hp length:sizeof(hp) atIndex:9];
                    [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:10];
                    [enc dispatchThreadgroups:MTLSizeMake(n_vh, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }

                if (il + 1u == c.n_layer && compute_mode == 0u) {
                    break;
                }
                buf_bar();
            } else {
                encode_turbo_mat(r, enc, L.w_qkv, r->x_rot_buf, r->x_rot_h_buf, 0, r->qkv_gqa_buf, 0, batch_m);
                if (L.has_split_gate != 0u) {
                    encode_turbo_mat(r, enc, L.w_gate_sep, r->x_rot_buf, r->x_rot_h_buf, 0, r->kv_gqa_buf, 0, batch_m);
                }
                buf_bar();

                uint32_t q_raw_dim = (L.gated_q ? 2u : 1u) * q_dim;
                uint32_t kv_stride = (L.has_split_gate != 0u) ? L.w_gate_sep.rows : L.w_qkv.rows;
                id<MTLBuffer> kv_b = (L.has_split_gate != 0u) ? r->kv_gqa_buf : r->qkv_gqa_buf;
                size_t kv_off = (L.has_split_gate != 0u) ? 0u : size_t(q_raw_dim) * sizeof(float);
                TurboGqaParamsC gp{
                    c.n_head, c.n_head_kv, c.head_dim, pos_base, c.rope_dim, L.gqa_flags,
                    L.w_qkv.rows, kv_stride, q_raw_dim, padded_qdim, batch_m, c.rope_freq_base, c.rms_norm_eps
                };

                if (batch_m == 1u && c.head_dim <= 256u && !(il + 1u == c.n_layer && compute_mode == 0u)) {
                    [enc setComputePipelineState:r->pso_gqa_fused_m1];
                    [enc setBuffer:r->qkv_gqa_buf offset:0 atIndex:0];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec0_off atIndex:1];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec1_off atIndex:2];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec2_off atIndex:3];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec3_off atIndex:4];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec4_off atIndex:5];
                    [enc setBuffer:s->k_caches[il] offset:0 atIndex:6];
                    [enc setBuffer:s->v_caches[il] offset:0 atIndex:7];
                    [enc setBuffer:kv_b offset:kv_off atIndex:8];
                    [enc setBuffer:r->x_rot_tmp_buf offset:0 atIndex:9];
                    [enc setBytes:&gp length:sizeof(gp) atIndex:10];
                    [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:11];
                    [enc dispatchThreadgroups:MTLSizeMake(c.n_head, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                } else {
                    [enc setComputePipelineState:r->pso_gqa_prep_fp16];
                    [enc setBuffer:r->qkv_gqa_buf offset:0 atIndex:0];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec0_off atIndex:1];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec1_off atIndex:2];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec2_off atIndex:3];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec3_off atIndex:4];
                    [enc setBuffer:r->weight_arena_buf offset:L.vec4_off atIndex:5];
                    [enc setBuffer:r->q_cur_buf offset:0 atIndex:6];
                    [enc setBuffer:r->q_gate_buf offset:0 atIndex:7];
                    [enc setBuffer:s->k_caches[il] offset:0 atIndex:8];
                    [enc setBuffer:s->v_caches[il] offset:0 atIndex:9];
                    [enc setBuffer:kv_b offset:kv_off atIndex:10];
                    [enc setBytes:&gp length:sizeof(gp) atIndex:11];
                    [enc dispatchThreadgroups:MTLSizeMake(c.n_head + c.n_head_kv, batch_m, 1)
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    if (il + 1u == c.n_layer && compute_mode == 0u) {
                        break;
                    }
                    buf_bar();

                    [enc setComputePipelineState:r->pso_gqa_attn_fp16];
                    [enc setBuffer:r->q_cur_buf offset:0 atIndex:0];
                    [enc setBuffer:r->q_gate_buf offset:0 atIndex:1];
                    [enc setBuffer:s->k_caches[il] offset:0 atIndex:2];
                    [enc setBuffer:s->v_caches[il] offset:0 atIndex:3];
                    [enc setBuffer:r->x_rot_tmp_buf offset:0 atIndex:4];
                    [enc setBytes:&gp length:sizeof(gp) atIndex:5];
                    [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(c.n_head, batch_m, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                buf_bar();
            }

            if (fuse_m1_res) {
                bool fuse_post = (can_epi_layer && run_ffn && L.w_out.padded_cols >= 512u && L.w_out.rows <= 16384u);
                encode_turbo_mat(
                    r, enc, L.w_out, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u,
                    true, fuse_post ? 1u : 0u, slot_post, L.post_attn_norm_off, padded_embd, false
                );
                post_norm_ready = fuse_post;
                buf_bar();
            } else {
                encode_turbo_mat(r, enc, L.w_out, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->attn_out_buf, 0, batch_m);
                buf_bar();
            }
        }

        if (run_ffn) {
            if (!fuse_m1_res) {
                TurboNormParamsC np{ dim, padded_embd, c.rms_norm_eps, batch_m };
                [enc setComputePipelineState:r->pso_add_rms_norm_turbo];
                [enc setBuffer:r->cur_buf offset:0 atIndex:0];
                [enc setBuffer:r->attn_out_buf offset:0 atIndex:1];
                [enc setBuffer:r->weight_arena_buf offset:L.post_attn_norm_off atIndex:2];
                [enc setBuffer:r->x_rot_buf offset:0 atIndex:3];
                [enc setBytes:&np length:sizeof(np) atIndex:4];
                [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1, batch_m, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                buf_bar();
            } else if (!post_norm_ready) {
                TurboNormParamsC np{ dim, padded_embd, c.rms_norm_eps, 1u | 256u };
                [enc setComputePipelineState:r->pso_rms_norm_turbo];
                [enc setBuffer:r->cur_buf offset:0 atIndex:0];
                [enc setBuffer:r->weight_arena_buf offset:L.post_attn_norm_off atIndex:1];
                [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
                [enc setBytes:&np length:sizeof(np) atIndex:3];
                [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                buf_bar();
            }

            if (can_fused_gu) {
                uint32_t ec_flag = (L.w_ffn_gu.bits == 4u) ? 256u : 0u;
                uint32_t gu_tgs = padded_ff >> 5u;
                TurboLlmMatParamsC gp{
                    L.w_ffn_gu.rows, L.w_ffn_gu.padded_cols, c.n_ff, ec_flag,
                    dim, c.rms_norm_eps, padded_ff, 0u, gu_tgs, 0u
                };
                id<MTLComputePipelineState> gu_pso = (L.w_ffn_gu.bits == 2u)
                    ? r->pso_m1_ffn_gu_t2
                    : ((L.w_ffn_gu.bits == 5u) ? r->pso_m1_ffn_gu_t25 : r->pso_m1_ffn_gu_t3);
                [enc setComputePipelineState:gu_pso];
                [enc setBuffer:r->weight_arena_buf offset:L.w_ffn_gu.scales_off atIndex:0];
                [enc setBuffer:r->weight_arena_buf offset:L.w_ffn_gu.packed_off atIndex:1];
                [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:2];
                [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:3];
                [enc setBytes:&gp length:sizeof(gp) atIndex:4];
                [enc setBuffer:r->lut_bank_buf offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(gu_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
                buf_bar();
            } else {
                bool fuse_sw = (can_epi_layer && L.w_ffn_gu.padded_cols >= 512u && L.w_ffn_gu.rows <= 16384u);
                encode_turbo_mat(
                    r, enc, L.w_ffn_gu, r->x_rot_buf, r->x_rot_h_buf, 0, r->ffn_gu_buf, 0, batch_m,
                    false, fuse_sw ? 2u : 0u, slot_gu, 0u, padded_ff, false
                );
                buf_bar();
                if (!fuse_sw) {
                    TurboSwigluParamsC sp{ c.n_ff, padded_ff, L.w_ffn_gu.rows, batch_m };
                    [enc setComputePipelineState:r->pso_swiglu_turbo];
                    [enc setBuffer:r->ffn_gu_buf offset:0 atIndex:0];
                    [enc setBuffer:r->x_rot_tmp_buf offset:0 atIndex:1];
                    [enc setBytes:&sp length:sizeof(sp) atIndex:2];
                    [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(padded_ff >> 8u, batch_m, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    buf_bar();
                }
            }

            if (fuse_m1_res) {
                bool has_nxt = (il + 1u < c.n_layer) && (((attn_mask >> (il + 1u)) & 1u) != 0u);
                bool fuse_nxt = (can_epi_layer && has_nxt && L.w_ffn_down.padded_cols >= 512u && L.w_ffn_down.rows <= 16384u);
                size_t nxt_norm_off = has_nxt ? r->layers[il + 1u].attn_norm_off : 0u;
                bool nxt_write_f = has_nxt && (r->layers[il + 1u].is_recurrent != 0u);
                encode_turbo_mat(
                    r, enc, L.w_ffn_down, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u,
                    true, fuse_nxt ? 1u : 0u, slot_next, nxt_norm_off, padded_embd, nxt_write_f
                );
                next_attn_norm_ready = fuse_nxt;
                buf_bar();
            } else {
                encode_turbo_mat(r, enc, L.w_ffn_down, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->ffn_out_buf, 0, batch_m);
                buf_bar();
                if (il + 1u < c.n_layer || compute_mode != 0u) {
                    size_t next_norm_off = (il + 1u < c.n_layer)
                        ? r->layers[il + 1u].attn_norm_off
                        : r->output_norm_off;
                    uint32_t next_pad = (il + 1u < c.n_layer) ? padded_embd : r->output_head_mat.padded_cols;
                    TurboNormParamsC np{ dim, next_pad, c.rms_norm_eps, batch_m };
                    [enc setComputePipelineState:r->pso_add_rms_norm_turbo];
                    [enc setBuffer:r->cur_buf offset:0 atIndex:0];
                    [enc setBuffer:r->ffn_out_buf offset:0 atIndex:1];
                    [enc setBuffer:r->weight_arena_buf offset:next_norm_off atIndex:2];
                    [enc setBuffer:r->x_rot_buf offset:0 atIndex:3];
                    [enc setBytes:&np length:sizeof(np) atIndex:4];
                    [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, batch_m, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    buf_bar();
                }
            }
        } else {
            next_attn_norm_ready = false;
        }
    }

    if (fuse_m1_res && compute_mode != 0u && !fuse_final_quant && !next_attn_norm_ready) {
        TurboNormParamsC np{ dim, r->output_head_mat.padded_cols, c.rms_norm_eps, 1u };
        [enc setComputePipelineState:r->pso_rms_norm_turbo];
        [enc setBuffer:r->cur_buf offset:0 atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:r->output_norm_off atIndex:1];
        [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
        [enc setBytes:&np length:sizeof(np) atIndex:3];
        [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        buf_bar();
    }

    // 4. LM Head / Argmax
    uint32_t d1 = 0xffffffffu, d2 = 0xffffffffu, d3 = 0xffffffffu;
    if (batch_m > 1u && tok_buf) {
        const auto* tin = reinterpret_cast<const uint32_t*>(
            static_cast<const uint8_t*>([tok_buf contents]) + tok_byte_off
        );
        d1 = tin[1];
        if (batch_m > 2u) d2 = tin[2];
        if (batch_m > 3u) d3 = tin[3];
    }
    if (compute_mode == 1u) {
        encode_turbo_mat(r, enc, r->output_head_mat, r->x_rot_buf, r->x_rot_h_buf, 0, r->logits_buf, 0, batch_m);
        buf_bar();
    } else if (compute_mode == 4u) {
        size_t x_elem_off = size_t(batch_m - 1u) * size_t(r->output_head_mat.padded_cols);
        encode_turbo_mat(r, enc, r->output_head_mat, r->x_rot_buf, r->x_rot_h_buf, x_elem_off, r->logits_buf, 0, 1u);
        buf_bar();
    } else if (compute_mode == 2u) {
        uint32_t pcols = r->output_head_mat.padded_cols;
        uint32_t n_vocab = c.n_vocab;
        uint32_t n_blocks64 = (n_vocab + 63u) / 64u;

        if (!final_quant_ready) {
            if (fuse_final_quant) {
                TurboNormParamsC np{ dim, pcols, c.rms_norm_eps, fuse_m1_res ? (1u | 256u) : 1u };
                [enc setComputePipelineState:r->pso_final_norm_quant];
                [enc setBuffer:r->cur_buf offset:0 atIndex:0];
                [enc setBuffer:r->ffn_out_buf offset:0 atIndex:1];
                [enc setBuffer:r->weight_arena_buf offset:r->output_norm_off atIndex:2];
                [enc setBuffer:r->x_rot_buf offset:0 atIndex:3];
                [enc setBuffer:r->x_planes_buf offset:0 atIndex:4];
                [enc setBuffer:r->x_qscale_buf offset:0 atIndex:5];
                [enc setBytes:&np length:sizeof(np) atIndex:6];
                [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                uint32_t qp[2] = { pcols, batch_m };
                [enc setComputePipelineState:r->pso_quant_x_sketch];
                [enc setBuffer:r->x_rot_buf offset:0 atIndex:0];
                [enc setBuffer:r->x_planes_buf offset:0 atIndex:1];
                [enc setBuffer:r->x_qscale_buf offset:0 atIndex:2];
                [enc setBytes:qp length:sizeof(qp) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, batch_m, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            buf_bar();
        }

        float eff_pen = use_bak_state ? 0.0f : s->presence_pen;
        TurboSketchEvalParamsC ep{ n_vocab, pcols, n_blocks64, r->stride_tiles8, eff_pen, d1, d2, d3 };
        [enc setComputePipelineState:r->pso_sketch_s1];
        [enc setBuffer:r->sketch_signs_buf offset:0 atIndex:0];
        [enc setBuffer:r->sketch_l1_buf offset:0 atIndex:1];
        [enc setBuffer:r->x_planes_buf offset:0 atIndex:2];
        [enc setBuffer:r->x_qscale_buf offset:0 atIndex:3];
        [enc setBuffer:r->blk_sketch_max_buf offset:0 atIndex:4];
        [enc setBytes:&ep length:sizeof(ep) atIndex:5];
        [enc setBuffer:s->seen_vocab_buf offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(r->n_tg_s1, batch_m, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        buf_bar();

        uint32_t tp[2] = { n_blocks64, batch_m };
        [enc setComputePipelineState:r->pso_sketch_thresh];
        [enc setBuffer:r->blk_sketch_max_buf offset:0 atIndex:0];
        [enc setBuffer:r->surv_blk_buf offset:0 atIndex:1];
        [enc setBytes:tp length:sizeof(tp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, batch_m, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        buf_bar();

        TurboSketchStage2ParamsC s2p{
            n_vocab, pcols, r->output_head_mat.bits, eff_pen, STAGE2_TGS, d1, d2, d3
        };
        id<MTLComputePipelineState> s2_pso = (r->output_head_mat.bits == 2u)
            ? r->pso_sketch_s2_t2
            : ((r->output_head_mat.bits == 5u) ? r->pso_sketch_s2_t25 : r->pso_sketch_s2_t3);
        [enc setComputePipelineState:s2_pso];
        [enc setBuffer:r->weight_arena_buf offset:r->output_head_mat.scales_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:r->output_head_mat.packed_off atIndex:1];
        [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:2];
        [enc setBuffer:r->surv_blk_buf offset:0 atIndex:3];
        [enc setBuffer:r->blk_vals_buf offset:0 atIndex:4];
        [enc setBuffer:r->blk_idxs_buf offset:0 atIndex:5];
        [enc setBytes:&s2p length:sizeof(s2p) atIndex:6];
        [enc setBuffer:s->seen_vocab_buf offset:0 atIndex:7];
        [enc setBuffer:r->lut_bank_buf offset:0 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(STAGE2_TGS, batch_m, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        buf_bar();

        TurboArgmaxParamsC ap{ n_vocab, STAGE2_TGS, batch_m, write_next_slot, eff_pen, d1, d2, d3 };
        [enc setComputePipelineState:r->pso_argmax_s2];
        [enc setBuffer:r->blk_vals_buf offset:0 atIndex:0];
        [enc setBuffer:r->blk_idxs_buf offset:0 atIndex:1];
        [enc setBuffer:out_ids_buf offset:out_ids_byte_off atIndex:2];
        [enc setBuffer:r->tok_ring_buf offset:0 atIndex:3];
        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
        [enc setBuffer:s->seen_vocab_buf offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, batch_m, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (write_next_slot != 0xffffffffu) buf_bar();
    } else if (compute_mode == 3u) {
        encode_turbo_mat(r, enc, r->output_head_mat, r->x_rot_buf, r->x_rot_h_buf, 0, r->logits_buf, 0, batch_m);
        buf_bar();
        uint32_t n_blocks1024 = (c.n_vocab + 1023u) / 1024u;
        TurboArgmaxParamsC ap{ c.n_vocab, n_blocks1024, batch_m, write_next_slot, s->presence_pen, d1, d2, d3 };

        [enc setComputePipelineState:r->pso_argmax_s1];
        [enc setBuffer:r->logits_buf offset:0 atIndex:0];
        [enc setBuffer:r->blk_vals_buf offset:0 atIndex:1];
        [enc setBuffer:r->blk_idxs_buf offset:0 atIndex:2];
        [enc setBytes:&ap length:sizeof(ap) atIndex:3];
        [enc setBuffer:s->seen_vocab_buf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(n_blocks1024, batch_m, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        buf_bar();

        [enc setComputePipelineState:r->pso_argmax_s2];
        [enc setBuffer:r->blk_vals_buf offset:0 atIndex:0];
        [enc setBuffer:r->blk_idxs_buf offset:0 atIndex:1];
        [enc setBuffer:out_ids_buf offset:out_ids_byte_off atIndex:2];
        [enc setBuffer:r->tok_ring_buf offset:0 atIndex:3];
        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
        [enc setBuffer:s->seen_vocab_buf offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, batch_m, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (write_next_slot != 0xffffffffu) buf_bar();
    }
}

} // namespace

static void run_vxt_timing_probe(MetalTurboRunner* r);

extern "C" {

void* vxt_metal_turbo_create(
    void* raw_gemv_ctx,
    const VxtLlmConfig* cfg,
    const VxtTurboMat* embed_mat,
    const VxtTurboLayerDesc* layer_descs,
    void* output_norm_buf,
    const VxtTurboMat* output_head_mat,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* src_ctx = static_cast<MetalGemvCtx*>(raw_gemv_ctx);
        if (!src_ctx || !cfg || !embed_mat || !layer_descs || !output_norm_buf || !output_head_mat) {
            vxt_write_err(err_out, err_cap, "null argument to vxt_metal_turbo_create");
            return nullptr;
        }
        auto* r = new MetalTurboRunner();
        r->gemv = *src_ctx;
        r->cfg = *cfg;
        r->has_coherent_dev = false;

        id<MTLDevice> dev = r->gemv.device;
        id<MTLLibrary> lib = r->gemv.fast_lib ? r->gemv.fast_lib : r->gemv.lib;

        auto need_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
            return vxt_build_pso(dev, lib, name, err_out, err_cap);
        };

        r->pso_build_lut_bank = need_pso(@"vxt_build_lut_bank");
        r->pso_embed_turbo = need_pso(@"vxt_embed_turbo");
        r->pso_embed_norm0_m1 = need_pso(@"vxt_embed_norm0_m1");
        r->pso_rms_norm_turbo = need_pso(@"vxt_rms_norm_turbo");
        r->pso_rms_norm_wait_m1 = need_pso(@"vxt_rms_norm_wait_m1");
        r->pso_add_rms_norm_turbo = need_pso(@"vxt_add_rms_norm_turbo");
        r->pso_swiglu_turbo = need_pso(@"vxt_swiglu_turbo");
        r->pso_llm_raw16_rot = need_pso(@"vxt_llm_raw16_rot");
        r->pso_m1_t2_1sg = need_pso(@"vxt_m1_t2_1sg");
        r->pso_m1_t2_sk2 = need_pso(@"vxt_m1_t2_sk2");
        r->pso_m1_t2_sk4 = need_pso(@"vxt_m1_t2_sk4");
        r->pso_m1_t25_1sg = need_pso(@"vxt_m1_t25_1sg");
        r->pso_m1_t25_sk2 = need_pso(@"vxt_m1_t25_sk2");
        r->pso_m1_t25_sk4 = need_pso(@"vxt_m1_t25_sk4");
        r->pso_m1_t3_1sg = need_pso(@"vxt_m1_t3_1sg");
        r->pso_m1_t3_sk2 = need_pso(@"vxt_m1_t3_sk2");
        r->pso_m1_t3_sk4 = need_pso(@"vxt_m1_t3_sk4");
        r->pso_m1_ffn_gu_t2 = need_pso(@"vxt_m1_ffn_gu_t2");
        r->pso_m1_ffn_gu_t25 = need_pso(@"vxt_m1_ffn_gu_t25");
        r->pso_m1_ffn_gu_t3 = need_pso(@"vxt_m1_ffn_gu_t3");
        r->pso_dnet_m1_prep16 = need_pso(@"vxt_dnet_m1_prep16");
        r->pso_dnet_m1_state64 = need_pso(@"vxt_dnet_m1_state64");
        r->pso_dnet_m1_out2 = need_pso(@"vxt_dnet_m1_out2");
        r->pso_dnet_m1_fused16 = need_pso(@"vxt_dnet_m1_fused16");
        r->pso_dnet_m1_fused64 = need_pso(@"vxt_dnet_m1_fused64");
        r->pso_llm_turbo2_gemm4 = need_pso(@"vxt_llm_turbo2_gemm4");
        r->pso_llm_turbo25_gemm4 = need_pso(@"vxt_llm_turbo25_gemm4");
        r->pso_llm_turbo3_gemm4 = need_pso(@"vxt_llm_turbo3_gemm4");
        r->pso_deltanet_conv_batch = need_pso(@"vxt_deltanet_conv_batch");
        r->pso_deltanet_fused_1to1 = need_pso(@"vxt_deltanet_fused_1to1_fp16");
        r->pso_deltanet_heads_fp16 = need_pso(@"vxt_deltanet_heads_fp16_turbo");
        r->pso_gqa_prep_fp16 = need_pso(@"vxt_gqa_prep_fp16_batch");
        r->pso_gqa_attn_fp16 = need_pso(@"vxt_gqa_attn_fp16_turbo");
        r->pso_gqa_fused_m1 = need_pso(@"vxt_gqa_fused_m1");
        r->pso_argmax_s1 = need_pso(@"vxt_argmax_stage1");
        r->pso_argmax_s2 = need_pso(@"vxt_argmax_stage2");
        r->pso_build_sketch = need_pso(@"vxt_build_lm_sketch");
        r->pso_quant_x_sketch = need_pso(@"vxt_quantize_x_sketch");
        r->pso_sketch_s1 = need_pso(@"vxt_lm_sketch_stage1");
        r->pso_sketch_thresh = need_pso(@"vxt_lm_sketch_threshold");
        r->pso_sketch_s2_t2 = need_pso(@"vxt_lm_exact_stage2_turbo2");
        r->pso_sketch_s2_t25 = need_pso(@"vxt_lm_exact_stage2_turbo25");
        r->pso_sketch_s2_t3 = need_pso(@"vxt_lm_exact_stage2_turbo3");
        r->pso_final_norm_quant = need_pso(@"vxt_final_norm_quant_sketch");
        r->pso_final_norm_quant_wait_m1 = need_pso(@"vxt_final_norm_quant_wait_m1");

        if (!r->pso_build_lut_bank || !r->pso_embed_turbo || !r->pso_embed_norm0_m1 || !r->pso_rms_norm_turbo ||
            !r->pso_rms_norm_wait_m1 || !r->pso_add_rms_norm_turbo ||
            !r->pso_swiglu_turbo || !r->pso_llm_raw16_rot ||
            !r->pso_m1_t2_1sg || !r->pso_m1_t2_sk2 || !r->pso_m1_t2_sk4 ||
            !r->pso_m1_t25_1sg || !r->pso_m1_t25_sk2 || !r->pso_m1_t25_sk4 ||
            !r->pso_m1_t3_1sg || !r->pso_m1_t3_sk2 || !r->pso_m1_t3_sk4 ||
            !r->pso_m1_ffn_gu_t2 || !r->pso_m1_ffn_gu_t25 || !r->pso_m1_ffn_gu_t3 ||
            !r->pso_dnet_m1_prep16 || !r->pso_dnet_m1_state64 || !r->pso_dnet_m1_out2 ||
            !r->pso_dnet_m1_fused16 || !r->pso_dnet_m1_fused64 ||
            !r->pso_llm_turbo2_gemm4 || !r->pso_llm_turbo25_gemm4 || !r->pso_llm_turbo3_gemm4 ||
            !r->pso_deltanet_conv_batch || !r->pso_deltanet_fused_1to1 || !r->pso_deltanet_heads_fp16 ||
            !r->pso_gqa_prep_fp16 || !r->pso_gqa_attn_fp16 || !r->pso_gqa_fused_m1 ||
            !r->pso_argmax_s1 || !r->pso_argmax_s2 ||
            !r->pso_build_sketch || !r->pso_quant_x_sketch || !r->pso_sketch_s1 ||
            !r->pso_sketch_thresh || !r->pso_sketch_s2_t2 || !r->pso_sketch_s2_t25 ||
            !r->pso_sketch_s2_t3 || !r->pso_final_norm_quant || !r->pso_final_norm_quant_wait_m1) {
            delete r;
            return nullptr;
        }
        r->can_dnet_fused16 = ([r->pso_dnet_m1_fused16 maxTotalThreadsPerThreadgroup] >= 512u);
        r->can_t25_sk2_512 = ([r->pso_m1_t25_sk4 maxTotalThreadsPerThreadgroup] >= 512u);
        r->can_t3_sk2_512 = ([r->pso_m1_t3_sk4 maxTotalThreadsPerThreadgroup] >= 512u);
        r->can_tg512_gu = ([r->pso_m1_ffn_gu_t2 maxTotalThreadsPerThreadgroup] >= 512u) &&
                          ([r->pso_m1_ffn_gu_t25 maxTotalThreadsPerThreadgroup] >= 512u) &&
                          ([r->pso_m1_ffn_gu_t3 maxTotalThreadsPerThreadgroup] >= 512u);

        // 1. Repack `Turbo3`/`EC-Turbo3` (6-bit row words) & `Turbo2.5` swizzle once in-place
        {
            std::vector<void*> seen_pw;
            auto swz_t25 = [](uint32_t c) -> uint32_t {
                return ((c >> 6u) & 3u) | ((c & 63u) << 2u) | (c & 0x300u);
            };
            auto repack_mat = [&](const VxtTurboMat& m) {
                if ((m.bits != 3u && m.bits != 4u && m.bits != 5u) || !m.packed_buf) return;
                for (void* p : seen_pw) if (p == m.packed_buf) return;
                seen_pw.push_back(m.packed_buf);
                uint32_t tiles = (m.rows + 3u) / 4u;
                uint32_t steps = m.padded_cols >> 8u;
                uint32_t* pw = static_cast<uint32_t*>([(__bridge id<MTLBuffer>)m.packed_buf contents]);
                size_t total_ts = size_t(tiles) * size_t(steps);
                if (m.bits == 5u) {
                    for (size_t ts = 0; ts < total_ts; ++ts) {
                        uint32_t* p80 = pw + ts * 80u;
                        uint16_t* h64 = reinterpret_cast<uint16_t*>(p80 + 64u);
                        for (uint32_t lane = 0; lane < 32u; ++lane) {
                            uint32_t w0 = p80[2u * lane], w1 = p80[2u * lane + 1u];
                            uint32_t h16 = h64[lane];
                            uint32_t s00 = swz_t25(w0 & 1023u);
                            uint32_t s01 = swz_t25((w0 >> 10u) & 1023u);
                            uint32_t s10 = swz_t25((w0 >> 20u) & 1023u);
                            uint32_t s11 = swz_t25(w1 & 1023u);
                            uint32_t s20 = swz_t25((w1 >> 10u) & 1023u);
                            uint32_t s21 = swz_t25((w1 >> 20u) & 1023u);
                            uint32_t s30 = swz_t25((h16 & 0xffu) | ((w0 >> 30u) << 8u));
                            uint32_t s31 = swz_t25((h16 >> 8u) | ((w1 >> 30u) << 8u));
                            p80[2u * lane]     = s00 | (s01 << 10u) | (s10 << 20u) | ((s30 >> 8u) << 30u);
                            p80[2u * lane + 1u] = s11 | (s20 << 10u) | (s21 << 20u) | ((s31 >> 8u) << 30u);
                            h64[lane]          = uint16_t((s30 & 0xffu) | ((s31 & 0xffu) << 8u));
                        }
                    }
                } else {
                    for (size_t ts = 0; ts < total_ts; ++ts) {
                        uint32_t* p96 = pw + ts * 96u;
                        for (uint32_t lane = 0; lane < 32u; ++lane) {
                            uint32_t iw0 = p96[2u * lane], iw1 = p96[2u * lane + 1u], sw = p96[64u + lane];
                            uint32_t idx16[4] = { iw0 & 0xffffu, iw0 >> 16u, iw1 & 0xffffu, iw1 >> 16u };
                            uint32_t c6[4][4];
                            for (uint32_t sr = 0; sr < 4u; ++sr) {
                                uint32_t sgn8 = (sw >> (sr * 8u)) & 0xffu;
                                for (uint32_t pr = 0; pr < 4u; ++pr) {
                                    c6[sr][pr] = ((idx16[sr] >> (pr * 4u)) & 0x0fu) | (((sgn8 >> (pr * 2u)) & 0x03u) << 4u);
                                }
                            }
                            p96[2u * lane]      = c6[0][0] | (c6[0][1] << 6u) | (c6[0][2] << 12u) | (c6[0][3] << 18u) | (c6[3][0] << 24u) | ((c6[3][3] & 0x03u) << 30u);
                            p96[2u * lane + 1u] = c6[1][0] | (c6[1][1] << 6u) | (c6[1][2] << 12u) | (c6[1][3] << 18u) | (c6[3][1] << 24u) | (((c6[3][3] >> 2u) & 0x03u) << 30u);
                            p96[64u + lane]     = c6[2][0] | (c6[2][1] << 6u) | (c6[2][2] << 12u) | (c6[2][3] << 18u) | (c6[3][2] << 24u) | (((c6[3][3] >> 4u) & 0x03u) << 30u);
                        }
                    }
                }
            };
            repack_mat(*embed_mat);
            repack_mat(*output_head_mat);
            for (uint32_t il = 0; il < cfg->n_layer; ++il) {
                const auto& L = layer_descs[il];
                repack_mat(L.w_qkv);
                repack_mat(L.w_gate_sep);
                repack_mat(L.w_out);
                repack_mat(L.w_ffn_gu);
                repack_mat(L.w_ffn_down);
            }
        }

        // 2. Pack all weights into ONE 256-byte aligned `weight_arena_buf` (offset 0 = 256 zeroed dummy bytes)
        size_t total_arena_bytes = 256u;
        std::vector<std::pair<void*, size_t>> arena_map;
        arena_map.reserve(cfg->n_layer * 16u + 8u);
        auto reg_buf = [&](void* raw) -> size_t {
            if (!raw) return 0u;
            for (const auto& kv : arena_map) {
                if (kv.first == raw) return kv.second;
            }
            size_t len = [(__bridge id<MTLBuffer>)raw length];
            size_t aligned = (len + 255u) & ~size_t(255u);
            size_t off = total_arena_bytes;
            total_arena_bytes += std::max<size_t>(aligned, 256u);
            arena_map.push_back({raw, off});
            return off;
        };
        auto reg_mat = [&](const VxtTurboMat& m) -> ArenaMat {
            ArenaMat am;
            am.scales_off = reg_buf(m.scales_buf);
            am.packed_off = reg_buf(m.packed_buf);
            am.rows = m.rows;
            am.cols = m.cols;
            am.padded_cols = m.padded_cols;
            am.bits = m.bits;
            return am;
        };

        r->embed_mat = reg_mat(*embed_mat);
        r->output_head_mat = reg_mat(*output_head_mat);
        r->output_norm_off = reg_buf(output_norm_buf);

        uint32_t max_rot = r->output_head_mat.padded_cols;
        uint32_t max_gqa_qkv = 16u;
        uint32_t max_gqa_kv = std::max<uint32_t>(2u * cfg->n_head_kv * cfg->head_dim, 16u);
        uint32_t max_gu = 2u * cfg->n_ff;

        r->layers.resize(cfg->n_layer);
        for (uint32_t il = 0; il < cfg->n_layer; ++il) {
            const auto& L = layer_descs[il];
            ArenaLayerDesc& AL = r->layers[il];
            AL.is_recurrent = L.is_recurrent;
            AL.gated_q = L.gated_q;
            AL.gqa_flags = L.gqa_flags;
            AL.has_split_gate = L.has_split_gate;
            AL.attn_norm_off = reg_buf(L.attn_norm_buf);
            AL.post_attn_norm_off = reg_buf(L.post_attn_norm_buf);
            AL.w_qkv = reg_mat(L.w_qkv);
            AL.w_gate_sep = reg_mat(L.w_gate_sep);
            AL.ab_rot_f16_off = reg_buf(L.ab_rot_f16_buf);
            AL.w_out = reg_mat(L.w_out);
            AL.vec0_off = reg_buf(L.vec0_buf);
            AL.vec1_off = reg_buf(L.vec1_buf);
            AL.vec2_off = reg_buf(L.vec2_buf);
            AL.vec3_off = reg_buf(L.vec3_buf);
            AL.vec4_off = reg_buf(L.vec4_buf);
            AL.w_ffn_gu = reg_mat(L.w_ffn_gu);
            AL.w_ffn_down = reg_mat(L.w_ffn_down);

            max_rot = std::max(max_rot, AL.w_qkv.padded_cols);
            max_rot = std::max(max_rot, AL.w_out.padded_cols);
            max_rot = std::max(max_rot, AL.w_ffn_gu.padded_cols);
            max_rot = std::max(max_rot, AL.w_ffn_down.padded_cols);
            if (!AL.is_recurrent) {
                max_gqa_qkv = std::max(max_gqa_qkv, AL.w_qkv.rows);
                if (AL.has_split_gate != 0u) max_gqa_kv = std::max(max_gqa_kv, AL.w_gate_sep.rows);
            }
            max_gu = std::max(max_gu, AL.w_ffn_gu.rows);
        }

        r->weight_arena_buf = [dev newBufferWithLength:total_arena_bytes options:MTLResourceStorageModeShared];
        if (!r->weight_arena_buf) {
            vxt_write_err(err_out, err_cap, "failed to allocate Metal weight_arena_buf");
            delete r;
            return nullptr;
        }
        {
            uint8_t* dst_base = static_cast<uint8_t*>([r->weight_arena_buf contents]);
            std::memset(dst_base, 0, 256u);
            for (const auto& kv : arena_map) {
                id<MTLBuffer> sb = (__bridge id<MTLBuffer>)kv.first;
                size_t len = [sb length];
                std::memcpy(dst_base + kv.second, [sb contents], len);
            }
        }

        uint32_t q_dim = std::max<uint32_t>(cfg->n_head * cfg->head_dim, 16u);
        uint32_t conv_ch = std::max<uint32_t>(
            2u * cfg->ssm_group_count * cfg->ssm_state_size + cfg->ssm_inner_size, 16u
        );
        uint32_t n_blocks64 = (cfg->n_vocab + 63u) / 64u;
        uint32_t n_tiles8 = n_blocks64 * 8u;
        r->n_tg_s1 = (n_tiles8 + 127u) / 128u;
        r->stride_tiles8 = ((n_tiles8 + 255u) / 256u) * 256u;
        uint32_t lm_pcols = std::max<uint32_t>(r->output_head_mat.padded_cols, 256u);
        uint32_t lm_steps = std::max<uint32_t>(lm_pcols >> 8u, 1u);
        r->tail_sync_base = cfg->n_layer * std::max<uint32_t>(cfg->ssm_dt_rank, 4u) + 8u;
        uint32_t total_sync_slots = std::max<uint32_t>(r->tail_sync_base + 3u * cfg->n_layer + 16u, 128u);
        uint32_t max_blk_vals = std::max<uint32_t>(STAGE2_TGS, (cfg->n_vocab + 1023u) / 1024u);

        r->lut_bank_buf = alloc_shared_bytes(dev, 16384);
        r->caps_buf = alloc_shared_bytes(dev, 64);
        r->cur_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * cfg->n_embd * sizeof(float));
        r->x_rot_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * max_rot * sizeof(float));
        r->x_rot_tmp_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * max_rot * sizeof(float));
        r->x_rot_h_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * max_rot * sizeof(uint16_t));
        r->x_rot_tmp_h_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * max_rot * sizeof(uint16_t));
        r->attn_out_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * cfg->n_embd * sizeof(float));
        r->ffn_gu_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * max_gu * sizeof(float));
        r->ffn_out_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * cfg->n_embd * sizeof(float));
        r->sync_cnt_buf = alloc_shared_bytes(dev, size_t(total_sync_slots) * sizeof(uint32_t));
        size_t dnet_tmp_floats = size_t(4u * cfg->ssm_dt_rank * 128u + 2u * cfg->ssm_dt_rank + 64u);
        size_t conv_silu_floats = std::max(size_t(MAX_BATCH_M) * size_t(conv_ch), dnet_tmp_floats);
        r->conv_silu_buf = alloc_shared_bytes(dev, conv_silu_floats * sizeof(float));
        r->qkv_gqa_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * max_gqa_qkv * sizeof(float));
        r->kv_gqa_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * max_gqa_kv * sizeof(float));
        r->q_cur_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * q_dim * sizeof(float));
        r->q_gate_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * q_dim * sizeof(float));
        r->logits_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * cfg->n_vocab * sizeof(float));

        r->sketch_signs_buf = alloc_shared_bytes(dev, size_t(lm_pcols) * size_t(r->stride_tiles8) * sizeof(uint16_t));
        r->sketch_l1_buf = alloc_shared_bytes(dev, size_t(lm_steps) * size_t(r->stride_tiles8) * 16u);
        r->x_planes_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * std::max<size_t>(size_t(lm_steps) * 256u * 8u, 4096u));
        r->x_qscale_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * std::max<size_t>((1u + size_t(lm_steps)) * sizeof(uint32_t), 1024u));
        r->blk_sketch_max_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * size_t(n_blocks64) * sizeof(float));
        r->surv_blk_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * 512u * sizeof(uint32_t));
        r->blk_vals_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * size_t(max_blk_vals) * sizeof(float));
        r->blk_idxs_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * size_t(max_blk_vals) * sizeof(uint32_t));

        r->tok_in_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * sizeof(uint32_t));
        r->out_ids_buf = alloc_shared_bytes(dev, size_t(MAX_BATCH_M) * sizeof(uint32_t));
        r->tok_ring_buf = alloc_shared_bytes(dev, size_t(RING_SLOTS) * sizeof(uint32_t));
        r->out_ring_buf = alloc_shared_bytes(dev, size_t(RING_SLOTS) * sizeof(uint32_t));

        if (!r->cur_buf || !r->x_rot_buf || !r->x_rot_tmp_buf || !r->x_rot_h_buf ||
            !r->x_rot_tmp_h_buf || !r->lut_bank_buf || !r->caps_buf || !r->kv_gqa_buf ||
            !r->sync_cnt_buf || !r->logits_buf || !r->sketch_signs_buf || !r->sketch_l1_buf ||
            !r->surv_blk_buf || !r->tok_ring_buf) {
            vxt_write_err(err_out, err_cap, "failed to allocate Metal Turbo buffers");
            delete r;
            return nullptr;
        }

        id<MTLCommandBuffer> cmd = [r->gemv.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:r->pso_build_lut_bank];
        [enc setBuffer:r->lut_bank_buf offset:0 atIndex:0];
        [enc setBuffer:r->caps_buf offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        TurboSketchBuildParamsC bp{
            r->output_head_mat.rows, r->output_head_mat.padded_cols, r->output_head_mat.bits, r->stride_tiles8
        };
        [enc setComputePipelineState:r->pso_build_sketch];
        [enc setBuffer:r->weight_arena_buf offset:r->output_head_mat.scales_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:r->output_head_mat.packed_off atIndex:1];
        [enc setBuffer:r->sketch_signs_buf offset:0 atIndex:2];
        [enc setBuffer:r->sketch_l1_buf offset:0 atIndex:3];
        [enc setBytes:&bp length:sizeof(bp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(r->stride_tiles8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            vxt_write_err(err_out, err_cap, "vxt_build_lm_sketch failed on GPU");
            delete r;
            return nullptr;
        }
        r->has_coherent_dev = (static_cast<const uint32_t*>([r->caps_buf contents])[0] != 0u);

        // Ramp GPU DVFS clock on full-size models so token-1 decode runs at boost frequency
        if (cfg->n_layer >= 16u && !std::getenv("VXT_TIMING")) {
            for (int w = 0; w < 4; ++w) {
                id<MTLCommandBuffer> wcb = [r->gemv.queue commandBuffer];
                id<MTLComputeCommandEncoder> wenc = [wcb computeCommandEncoder];
                for (int rep = 0; rep < 6; ++rep) {
                    for (uint32_t il = 0; il < cfg->n_layer; ++il) {
                        const auto& L = r->layers[il];
                        encode_turbo_mat(r, wenc, L.w_ffn_gu, r->x_rot_buf, r->x_rot_h_buf, 0, r->ffn_gu_buf, 0, 1u);
                        encode_turbo_mat(r, wenc, L.w_ffn_down, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->ffn_out_buf, 0, 1u);
                    }
                }
                [wenc endEncoding];
                [wcb commit];
                [wcb waitUntilCompleted];
            }
        }

        if (std::getenv("VXT_TIMING")) run_vxt_timing_probe(r);
        return r;
    }
}

void* vxt_metal_turbo_state_create(void* raw_runner, char* err_out, size_t err_cap);
void vxt_metal_turbo_state_reset(void* raw_state);
void vxt_metal_turbo_state_destroy(void* raw_state);

static void run_vxt_timing_probe(MetalTurboRunner* r) {
    const VxtLlmConfig& c = r->cfg;
    uint32_t cnt_b2 = 0, cnt_b34 = 0, cnt_b5 = 0, cnt_split = 0;
    auto add_mat = [&](const ArenaMat& m) {
        if (m.rows == 0u) return;
        if (m.bits == 2u) cnt_b2++;
        else if (m.bits == 5u) cnt_b5++;
        else cnt_b34++;
    };
    for (uint32_t il = 0; il < c.n_layer; ++il) {
        const auto& L = r->layers[il];
        add_mat(L.w_qkv);
        if (L.has_split_gate) { cnt_split++; add_mat(L.w_gate_sep); }
        add_mat(L.w_out);
        add_mat(L.w_ffn_gu);
        add_mat(L.w_ffn_down);
    }
    std::fprintf(
        stderr,
        "[VXT_TIMING] coh=%d dnet512=%d gu_512=%d | mats: b2=%u b25=%u b3/4=%u split=%u | arena=%.2f MiB | head=%u\n",
        int(r->has_coherent_dev), int(r->can_dnet_fused16), int(r->can_tg512_gu),
        cnt_b2, cnt_b5, cnt_b34, cnt_split,
        double([r->weight_arena_buf length]) / (1024.0 * 1024.0), r->output_head_mat.bits
    );

    char err[256] = {0};
    auto* ts = static_cast<MetalTurboState*>(vxt_metal_turbo_state_create(r, err, sizeof(err)));
    if (!ts) return;
    vxt_metal_turbo_state_reset(ts);
    ts->ensure_kv_cap(512u);

    static_cast<uint32_t*>([r->tok_ring_buf contents])[0] = 42u;
    for (int w = 0; w < 10; ++w) {
        id<MTLCommandBuffer> cb = [r->gemv.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (uint32_t k = 0; k < 8u; ++k) {
            encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u + k, 0u, 2u, r->out_ring_buf, 0, 1u);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    auto measure_gpu_us = [&](const char* label, auto encode_fn) {
        double best_us = 1e18, sum_us = 0.0;
        const int WARM = 12, ITERS = 20;
        for (int it = 0; it < WARM + ITERS; ++it) {
            id<MTLCommandBuffer> cb = [r->gemv.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode_fn(enc);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (it >= WARM) {
                double us = ([cb GPUEndTime] - [cb GPUStartTime]) * 1e6;
                if (us < best_us) best_us = us;
                sum_us += us;
            }
        }
        std::fprintf(stderr, "[VXT_TIMING] %-36s : best %7.1f us | avg %7.1f us\n", label, best_us, sum_us / ITERS);
    };

    uint32_t padded_embd = ((c.n_embd + 255u) / 256u) * 256u;
    uint32_t n_blocks64 = (c.n_vocab + 63u) / 64u;

    measure_gpu_us("A1. 96 GEMVs single-arena pure", [&](id<MTLComputeCommandEncoder> enc) {
        for (uint32_t il = 0; il < c.n_layer; ++il) {
            const auto& L = r->layers[il];
            id<MTLBuffer> qkv_out = L.is_recurrent ? ts->qkv_hist[il] : r->qkv_gqa_buf;
            encode_turbo_mat(r, enc, L.w_qkv, r->x_rot_buf, r->x_rot_h_buf, 0, qkv_out, 0, 1u);
            if (L.has_split_gate) {
                id<MTLBuffer> g_out = L.is_recurrent ? ts->z_sep_hist[il] : r->kv_gqa_buf;
                encode_turbo_mat(r, enc, L.w_gate_sep, r->x_rot_buf, r->x_rot_h_buf, 0, g_out, 0, 1u);
            }
            encode_turbo_mat(r, enc, L.w_out, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u, true);
            encode_turbo_mat(r, enc, L.w_ffn_gu, r->x_rot_buf, r->x_rot_h_buf, 0, r->ffn_gu_buf, 0, 1u);
            encode_turbo_mat(r, enc, L.w_ffn_down, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u, true);
        }
    });
    measure_gpu_us("A1_m4. 96 GEMMs M=4 pure", [&](id<MTLComputeCommandEncoder> enc) {
        for (uint32_t il = 0; il < c.n_layer; ++il) {
            const auto& L = r->layers[il];
            id<MTLBuffer> qkv_out = L.is_recurrent ? ts->qkv_hist[il] : r->qkv_gqa_buf;
            encode_turbo_mat(r, enc, L.w_qkv, r->x_rot_buf, r->x_rot_h_buf, 0, qkv_out, 0, 4u);
            if (L.has_split_gate) {
                id<MTLBuffer> g_out = L.is_recurrent ? ts->z_sep_hist[il] : r->kv_gqa_buf;
                encode_turbo_mat(r, enc, L.w_gate_sep, r->x_rot_buf, r->x_rot_h_buf, 0, g_out, 0, 4u);
            }
            encode_turbo_mat(r, enc, L.w_out, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->attn_out_buf, 0, 4u);
            encode_turbo_mat(r, enc, L.w_ffn_gu, r->x_rot_buf, r->x_rot_h_buf, 0, r->ffn_gu_buf, 0, 4u);
            encode_turbo_mat(r, enc, L.w_ffn_down, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->ffn_out_buf, 0, 4u);
        }
    });

    {
        double best_c = 1e18;
        for (int it = 0; it < 24; ++it) {
            id<MTLCommandBuffer> cb = [r->gemv.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            for (uint32_t il = 0; il < c.n_layer; ++il) {
                const auto& L = r->layers[il];
                id<MTLBuffer> qkv_out = L.is_recurrent ? ts->qkv_hist[il] : r->qkv_gqa_buf;
                encode_turbo_mat(r, enc, L.w_qkv, r->x_rot_buf, r->x_rot_h_buf, 0, qkv_out, 0, 1u);
                encode_turbo_mat(r, enc, L.w_out, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u, false);
                encode_turbo_mat(r, enc, L.w_ffn_gu, r->x_rot_buf, r->x_rot_h_buf, 0, r->ffn_gu_buf, 0, 1u);
                encode_turbo_mat(r, enc, L.w_ffn_down, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->ffn_out_buf, 0, 1u, false);
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (it >= 8) best_c = std::min(best_c, ([cb GPUEndTime] - [cb GPUStartTime]) * 1e6);
        }
        std::fprintf(stderr, "[VXT_TIMING] A1_conc. 96 GEMVs ZERO-BARRIER conc  : best %7.1f us\n", best_c);
        double best_cb = 1e30;
        for (int it = 0; it < 20; ++it) {
            id<MTLCommandBuffer> cb = [r->gemv.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            for (uint32_t il = 0; il < c.n_layer; ++il) {
                const auto& L = r->layers[il];
                id<MTLBuffer> qkv_out = L.is_recurrent ? ts->qkv_hist[il] : r->qkv_gqa_buf;
                encode_turbo_mat(r, enc, L.w_qkv, r->x_rot_buf, r->x_rot_h_buf, 0, qkv_out, 0, 1u);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_turbo_mat(r, enc, L.w_out, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u, false);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_turbo_mat(r, enc, L.w_ffn_gu, r->x_rot_buf, r->x_rot_h_buf, 0, r->ffn_gu_buf, 0, 1u);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_turbo_mat(r, enc, L.w_ffn_down, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->ffn_out_buf, 0, 1u, false);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (it >= 6) best_cb = std::min(best_cb, ([cb GPUEndTime] - [cb GPUStartTime]) * 1e6);
        }
        std::fprintf(stderr, "[VXT_TIMING] A1_conc_bar. 96 GEMVs + 96 BufBar    : best %7.1f us\n", best_cb);
    }

    uint32_t padded_ff = ((c.n_ff + 255u) / 256u) * 256u;
    measure_gpu_us("B1. 47x rms_norm + 24x swiglu pure", [&](id<MTLComputeCommandEncoder> enc) {
        for (uint32_t il = 0; il < c.n_layer; ++il) {
            const auto& L = r->layers[il];
            TurboNormParamsC np{ c.n_embd, padded_embd, c.rms_norm_eps, 1u };
            [enc setComputePipelineState:r->pso_rms_norm_turbo];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:L.attn_norm_off atIndex:1];
            [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc setComputePipelineState:r->pso_rms_norm_turbo];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:L.post_attn_norm_off atIndex:1];
            [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            TurboSwigluParamsC sp{ c.n_ff, padded_ff, L.w_ffn_gu.rows, 1u };
            [enc setComputePipelineState:r->pso_swiglu_turbo];
            [enc setBuffer:r->ffn_gu_buf offset:0 atIndex:0];
            [enc setBuffer:r->x_rot_tmp_buf offset:0 atIndex:1];
            [enc setBytes:&sp length:sizeof(sp) atIndex:2];
            [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(padded_ff >> 8u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    });
    measure_gpu_us("B2. A1 + 47x norm + 24x swiglu", [&](id<MTLComputeCommandEncoder> enc) {
        for (uint32_t il = 0; il < c.n_layer; ++il) {
            const auto& L = r->layers[il];
            TurboNormParamsC np{ c.n_embd, padded_embd, c.rms_norm_eps, 1u };
            [enc setComputePipelineState:r->pso_rms_norm_turbo];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:L.attn_norm_off atIndex:1];
            [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            id<MTLBuffer> qkv_out = L.is_recurrent ? ts->qkv_hist[il] : r->qkv_gqa_buf;
            encode_turbo_mat(r, enc, L.w_qkv, r->x_rot_buf, r->x_rot_h_buf, 0, qkv_out, 0, 1u);
            encode_turbo_mat(r, enc, L.w_out, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u, true);

            [enc setComputePipelineState:r->pso_rms_norm_turbo];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:L.post_attn_norm_off atIndex:1];
            [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_turbo_mat(r, enc, L.w_ffn_gu, r->x_rot_buf, r->x_rot_h_buf, 0, r->ffn_gu_buf, 0, 1u);

            TurboSwigluParamsC sp{ c.n_ff, padded_ff, L.w_ffn_gu.rows, 1u };
            [enc setComputePipelineState:r->pso_swiglu_turbo];
            [enc setBuffer:r->ffn_gu_buf offset:0 atIndex:0];
            [enc setBuffer:r->x_rot_tmp_buf offset:0 atIndex:1];
            [enc setBytes:&sp length:sizeof(sp) atIndex:2];
            [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(padded_ff >> 8u, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_turbo_mat(r, enc, L.w_ffn_down, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u, true);
        }
    });

    measure_gpu_us("B3. A1 + 47x norm + lean 512t ffn_gu", [&](id<MTLComputeCommandEncoder> enc) {
        for (uint32_t il = 0; il < c.n_layer; ++il) {
            const auto& L = r->layers[il];
            TurboNormParamsC np{ c.n_embd, padded_embd, c.rms_norm_eps, 1u };
            [enc setComputePipelineState:r->pso_rms_norm_turbo];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:L.attn_norm_off atIndex:1];
            [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            id<MTLBuffer> qkv_out = L.is_recurrent ? ts->qkv_hist[il] : r->qkv_gqa_buf;
            encode_turbo_mat(r, enc, L.w_qkv, r->x_rot_buf, r->x_rot_h_buf, 0, qkv_out, 0, 1u);
            encode_turbo_mat(r, enc, L.w_out, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u, true);

            [enc setComputePipelineState:r->pso_rms_norm_turbo];
            [enc setBuffer:r->cur_buf offset:0 atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:L.post_attn_norm_off atIndex:1];
            [enc setBuffer:r->x_rot_buf offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            uint32_t ec_flag = (L.w_ffn_gu.bits == 4u) ? 256u : 0u;
            uint32_t gu_tgs = padded_ff >> 5u;
            TurboLlmMatParamsC gp{
                L.w_ffn_gu.rows, L.w_ffn_gu.padded_cols, c.n_ff, ec_flag,
                c.n_embd, c.rms_norm_eps, padded_ff, 0u, gu_tgs, 0u
            };
            id<MTLComputePipelineState> gu_pso = (L.w_ffn_gu.bits == 2u)
                ? r->pso_m1_ffn_gu_t2
                : ((L.w_ffn_gu.bits == 5u) ? r->pso_m1_ffn_gu_t25 : r->pso_m1_ffn_gu_t3);
            [enc setComputePipelineState:gu_pso];
            [enc setBuffer:r->weight_arena_buf offset:L.w_ffn_gu.scales_off atIndex:0];
            [enc setBuffer:r->weight_arena_buf offset:L.w_ffn_gu.packed_off atIndex:1];
            [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:2];
            [enc setBuffer:r->x_rot_tmp_h_buf offset:0 atIndex:3];
            [enc setBytes:&gp length:sizeof(gp) atIndex:4];
            [enc setBuffer:r->lut_bank_buf offset:0 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(gu_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];

            encode_turbo_mat(r, enc, L.w_ffn_down, r->x_rot_tmp_buf, r->x_rot_tmp_h_buf, 0, r->cur_buf, 0, 1u, true);
        }
    });

    measure_gpu_us("E1. New 2-Bit Ind Sketch S1+Thresh", [&](id<MTLComputeCommandEncoder> enc) {
        TurboSketchEvalParamsC ep{
            c.n_vocab, padded_embd, n_blocks64, r->stride_tiles8, 0.0f, 0xffffffffu, 0xffffffffu, 0xffffffffu
        };
        [enc setComputePipelineState:r->pso_sketch_s1];
        [enc setBuffer:r->sketch_signs_buf offset:0 atIndex:0];
        [enc setBuffer:r->sketch_l1_buf offset:0 atIndex:1];
        [enc setBuffer:r->x_planes_buf offset:0 atIndex:2];
        [enc setBuffer:r->x_qscale_buf offset:0 atIndex:3];
        [enc setBuffer:r->blk_sketch_max_buf offset:0 atIndex:4];
        [enc setBytes:&ep length:sizeof(ep) atIndex:5];
        [enc setBuffer:r->weight_arena_buf offset:0 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(r->n_tg_s1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        uint32_t tp[2] = { n_blocks64, 1u };
        [enc setComputePipelineState:r->pso_sketch_thresh];
        [enc setBuffer:r->blk_sketch_max_buf offset:0 atIndex:0];
        [enc setBuffer:r->surv_blk_buf offset:0 atIndex:1];
        [enc setBytes:tp length:sizeof(tp) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });

    measure_gpu_us("E2. New Dense 448-TG Stage2+Argmax", [&](id<MTLComputeCommandEncoder> enc) {
        TurboSketchStage2ParamsC s2p{
            c.n_vocab, padded_embd, r->output_head_mat.bits, 0.0f, STAGE2_TGS, 0xffffffffu, 0xffffffffu, 0xffffffffu
        };
        id<MTLComputePipelineState> s2_pso = (r->output_head_mat.bits == 2u)
            ? r->pso_sketch_s2_t2
            : ((r->output_head_mat.bits == 5u) ? r->pso_sketch_s2_t25 : r->pso_sketch_s2_t3);
        [enc setComputePipelineState:s2_pso];
        [enc setBuffer:r->weight_arena_buf offset:r->output_head_mat.scales_off atIndex:0];
        [enc setBuffer:r->weight_arena_buf offset:r->output_head_mat.packed_off atIndex:1];
        [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:2];
        [enc setBuffer:r->surv_blk_buf offset:0 atIndex:3];
        [enc setBuffer:r->blk_vals_buf offset:0 atIndex:4];
        [enc setBuffer:r->blk_idxs_buf offset:0 atIndex:5];
        [enc setBytes:&s2p length:sizeof(s2p) atIndex:6];
        [enc setBuffer:ts->seen_vocab_buf offset:0 atIndex:7];
        [enc setBuffer:r->lut_bank_buf offset:0 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(STAGE2_TGS, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        TurboArgmaxParamsC ap{ c.n_vocab, STAGE2_TGS, 1u, 1u, 0.0f, 0xffffffffu, 0xffffffffu, 0xffffffffu };
        [enc setComputePipelineState:r->pso_argmax_s2];
        [enc setBuffer:r->blk_vals_buf offset:0 atIndex:0];
        [enc setBuffer:r->blk_idxs_buf offset:0 atIndex:1];
        [enc setBuffer:r->out_ring_buf offset:0 atIndex:2];
        [enc setBuffer:r->tok_ring_buf offset:0 atIndex:3];
        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
        [enc setBuffer:ts->seen_vocab_buf offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });

    measure_gpu_us("C3a. Trunk (dnet_fuse=1)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u, 0u, 0u, r->out_ring_buf, 0, 1u, 0xffffffffu, 0xffffffffu, false, 1);
    });
    measure_gpu_us("C3b. Trunk (dnet_fuse=0)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u, 0u, 0u, r->out_ring_buf, 0, 1u, 0xffffffffu, 0xffffffffu, false, 0);
    });
    measure_gpu_us("C3c. Trunk (dnet_fused16=2)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u, 0u, 0u, r->out_ring_buf, 0, 1u, 0xffffffffu, 0xffffffffu, false, 2);
    });
    measure_gpu_us("C3d. Trunk (dnet_fused64=3)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u, 0u, 0u, r->out_ring_buf, 0, 1u, 0xffffffffu, 0xffffffffu, false, 3);
    });
    measure_gpu_us("C4a. Full step mode=2 (dnet_fuse=1)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u, 0u, 2u, r->out_ring_buf, 0, 1u, 0xffffffffu, 0xffffffffu, false, 1);
    });
    measure_gpu_us("C4c. Full step mode=2 (dnet_fused16=2)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u, 0u, 2u, r->out_ring_buf, 0, 1u, 0xffffffffu, 0xffffffffu, false, 2);
    });
    measure_gpu_us("C4d. Full step mode=2 (dnet_fused64=3)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u, 0u, 2u, r->out_ring_buf, 0, 1u, 0xffffffffu, 0xffffffffu, false, 3);
    });
    measure_gpu_us("C5. Burst-8 mode=2 (8 tok total)", [&](id<MTLComputeCommandEncoder> enc) {
        for (uint32_t k = 0; k < 8u; ++k) {
            uint32_t sl = k & (RING_SLOTS - 1u);
            uint32_t nx = (k + 1u) & (RING_SLOTS - 1u);
            encode_turbo_forward(r, ts, enc, r->tok_ring_buf, sl * 4u, 1u, 64u + k, 0u, 2u, r->out_ring_buf, sl * 4u, nx);
        }
    });
    measure_gpu_us("C6. Burst-16 mode=2 (16 tok total)", [&](id<MTLComputeCommandEncoder> enc) {
        for (uint32_t k = 0; k < 16u; ++k) {
            uint32_t sl = k & (RING_SLOTS - 1u);
            uint32_t nx = (k + 1u) & (RING_SLOTS - 1u);
            encode_turbo_forward(r, ts, enc, r->tok_ring_buf, sl * 4u, 1u, 64u + k, 0u, 2u, r->out_ring_buf, sl * 4u, nx);
        }
    });
    measure_gpu_us("C7. Batch M=4 trunk (4 tok total)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_in_buf, 0, 4u, 64u, 1u, 0u, r->out_ids_buf, 0, 0xffffffffu);
    });
    measure_gpu_us("C8. Batch M=4 mode=3 (4 tok total)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_in_buf, 0, 4u, 64u, 1u, 3u, r->out_ids_buf, 0, 0xffffffffu);
    });
    measure_gpu_us("C8b. Batch M=4 mode=2 (4 tok total)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_in_buf, 0, 4u, 64u, 1u, 2u, r->out_ids_buf, 0, 0xffffffffu);
    });
    measure_gpu_us("C8c. Batch M=2 mode=2 (2 tok total)", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_in_buf, 0, 2u, 64u, 1u, 2u, r->out_ids_buf, 0, 0xffffffffu);
    });
    measure_gpu_us("C9. Draft 6-layer M=1 mode=2", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u, 0u, 2u, r->out_ring_buf, 0, 1u, 0x00888889u, 0x00888889u, true);
    });
    measure_gpu_us("C9b. Draft 4-layer M=1 mode=2", [&](id<MTLComputeCommandEncoder> enc) {
        encode_turbo_forward(r, ts, enc, r->tok_ring_buf, 0, 1u, 64u, 0u, 2u, r->out_ring_buf, 0, 1u, 0x00808081u, 0x00808081u, true);
    });

    vxt_metal_turbo_state_destroy(ts);
}

void vxt_metal_turbo_destroy(void* raw_runner) {
    @autoreleasepool {
        delete static_cast<MetalTurboRunner*>(raw_runner);
    }
}

void* vxt_metal_turbo_state_create(void* raw_runner, char* err_out, size_t err_cap) {
    @autoreleasepool {
        auto* r = static_cast<MetalTurboRunner*>(raw_runner);
        if (!r) {
            vxt_write_err(err_out, err_cap, "null MetalTurboRunner");
            return nullptr;
        }
        const VxtLlmConfig& c = r->cfg;
        auto* s = new MetalTurboState();
        s->device = r->gemv.device;
        s->n_layer = c.n_layer;
        s->n_vocab = c.n_vocab;
        s->kv_stride = c.n_head_kv * c.head_dim;
        s->cap_seq = 1024u;
        s->seen_vocab_buf = alloc_shared_bytes(s->device, size_t(c.n_vocab));
        if (!s->seen_vocab_buf) {
            vxt_write_err(err_out, err_cap, "failed to allocate Turbo seen_vocab buffer");
            delete s;
            return nullptr;
        }
        s->is_rec.resize(c.n_layer, 0);
        s->conv_bytes.resize(c.n_layer, 0);
        s->ssm_bytes.resize(c.n_layer, 0);
        s->conv_states.resize(c.n_layer, nil);
        s->conv_bak.resize(c.n_layer, nil);
        s->ssm_states.resize(c.n_layer, nil);
        s->ssm_bak.resize(c.n_layer, nil);
        s->qkv_hist.resize(c.n_layer, nil);
        s->z_sep_hist.resize(c.n_layer, nil);
        s->ab_hist.resize(c.n_layer, nil);
        s->k_caches.resize(c.n_layer, nil);
        s->v_caches.resize(c.n_layer, nil);

        uint32_t conv_ch = 2u * c.ssm_group_count * c.ssm_state_size + c.ssm_inner_size;
        uint32_t hist = c.ssm_conv_kernel > 0 ? (c.ssm_conv_kernel - 1u) : 0u;
        uint32_t dv = c.ssm_dt_rank > 0 ? (c.ssm_inner_size / c.ssm_dt_rank) : 0u;

        for (uint32_t il = 0; il < c.n_layer; ++il) {
            const auto& L = r->layers[il];
            if (L.is_recurrent != 0u) {
                s->is_rec[il] = 1;
                size_t cb = size_t(hist) * size_t(conv_ch) * sizeof(float);
                size_t sb = size_t(c.ssm_dt_rank) * size_t(dv) * size_t(c.ssm_state_size) * sizeof(uint16_t);
                s->conv_bytes[il] = cb;
                s->ssm_bytes[il] = sb;
                s->conv_states[il] = alloc_shared_bytes(s->device, cb);
                s->conv_bak[il] = alloc_shared_bytes(s->device, cb);
                s->ssm_states[il] = alloc_shared_bytes(s->device, sb);
                s->ssm_bak[il] = alloc_shared_bytes(s->device, sb);
                s->qkv_hist[il] = alloc_shared_bytes(s->device, size_t(MAX_BATCH_M) * L.w_qkv.rows * sizeof(float));
                if (L.has_split_gate != 0u) {
                    s->z_sep_hist[il] = alloc_shared_bytes(
                        s->device, size_t(MAX_BATCH_M) * L.w_gate_sep.rows * sizeof(float)
                    );
                }
                s->ab_hist[il] = alloc_shared_bytes(
                    s->device, size_t(MAX_BATCH_M) * 2u * c.ssm_dt_rank * sizeof(float)
                );
                if (!s->conv_states[il] || !s->conv_bak[il] || !s->ssm_states[il] ||
                    !s->ssm_bak[il] || !s->qkv_hist[il] || !s->ab_hist[il]) {
                    vxt_write_err(err_out, err_cap, "failed to allocate Turbo recurrent state");
                    delete s;
                    return nullptr;
                }
            } else {
                size_t kvb = size_t(s->cap_seq) * size_t(s->kv_stride) * sizeof(uint16_t);
                s->k_caches[il] = alloc_shared_bytes(s->device, kvb);
                s->v_caches[il] = alloc_shared_bytes(s->device, kvb);
                if (!s->k_caches[il] || !s->v_caches[il]) {
                    vxt_write_err(err_out, err_cap, "failed to allocate Turbo FP16 KV cache");
                    delete s;
                    return nullptr;
                }
            }
        }
        return s;
    }
}

void vxt_metal_turbo_state_reset(void* raw_state) {
    @autoreleasepool {
        auto* s = static_cast<MetalTurboState*>(raw_state);
        if (!s) return;
        s->presence_pen = 0.0f;
        if (s->seen_vocab_buf && s->n_vocab > 0) {
            std::memset([s->seen_vocab_buf contents], 0, s->n_vocab);
        }
        for (uint32_t il = 0; il < s->n_layer; ++il) {
            if (s->is_rec[il]) {
                if (s->conv_bytes[il] > 0) std::memset([s->conv_states[il] contents], 0, s->conv_bytes[il]);
                if (s->ssm_bytes[il] > 0) std::memset([s->ssm_states[il] contents], 0, s->ssm_bytes[il]);
            }
        }
    }
}

void vxt_metal_turbo_init_seen(
    void* raw_state,
    const uint32_t* seen_ids,
    size_t n_seen,
    float presence_pen
) {
    auto* s = static_cast<MetalTurboState*>(raw_state);
    if (!s || !s->seen_vocab_buf) return;
    s->presence_pen = presence_pen;
    auto* ptr = static_cast<uint8_t*>([s->seen_vocab_buf contents]);
    std::memset(ptr, 0, s->n_vocab);
    if (presence_pen > 0.0f && seen_ids) {
        for (size_t i = 0; i < n_seen; ++i) {
            uint32_t tid = seen_ids[i];
            if (tid < s->n_vocab && ptr[tid] < 255u) ptr[tid] += 1u;
        }
    }
}

void vxt_metal_turbo_mark_seen(void* raw_state, uint32_t tok_id) {
    auto* s = static_cast<MetalTurboState*>(raw_state);
    if (!s || !s->seen_vocab_buf || s->presence_pen <= 0.0f) return;
    if (tok_id < s->n_vocab) {
        auto* ptr = static_cast<uint8_t*>([s->seen_vocab_buf contents]);
        if (ptr[tok_id] < 255u) ptr[tok_id] += 1u;
    }
}

void vxt_metal_turbo_state_destroy(void* raw_state) {
    @autoreleasepool {
        delete static_cast<MetalTurboState*>(raw_state);
    }
}

int vxt_metal_turbo_step_batch(
    void* raw_runner,
    void* raw_state,
    const uint32_t* tok_ids,
    uint32_t batch_m,
    uint32_t pos_base,
    uint32_t save_backup,
    uint32_t compute_mode,
    float* logits_out,
    uint32_t* argmax_ids_out,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* r = static_cast<MetalTurboRunner*>(raw_runner);
        auto* s = static_cast<MetalTurboState*>(raw_state);
        if (!r || !s || !tok_ids || batch_m == 0u || batch_m > MAX_BATCH_M) {
            vxt_write_err(err_out, err_cap, "invalid args to vxt_metal_turbo_step_batch");
            return -1;
        }
        if (!s->ensure_kv_cap(pos_base + batch_m)) {
            vxt_write_err(err_out, err_cap, "failed to grow Turbo KV cache");
            return -2;
        }
        std::memcpy([r->tok_in_buf contents], tok_ids, size_t(batch_m) * sizeof(uint32_t));

        id<MTLCommandBuffer> cmd = [r->gemv.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            vxt_write_err(err_out, err_cap, "failed to create Metal command buffer/encoder");
            return -3;
        }

        encode_turbo_forward(
            r, s, enc, r->tok_in_buf, 0, batch_m, pos_base, save_backup,
            compute_mode, r->out_ids_buf, 0, 0xffffffffu
        );
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            NSString* desc = [[cmd error] localizedDescription];
            vxt_write_err(err_out, err_cap, desc ? [desc UTF8String] : "Turbo step_batch failed");
            return -4;
        }

        if (logits_out) {
            size_t count = (compute_mode == 4u) ? size_t(r->cfg.n_vocab) : size_t(batch_m) * size_t(r->cfg.n_vocab);
            std::memcpy(logits_out, [r->logits_buf contents], count * sizeof(float));
        }
        if (argmax_ids_out && (compute_mode == 2u || compute_mode == 3u)) {
            std::memcpy(argmax_ids_out, [r->out_ids_buf contents], size_t(batch_m) * sizeof(uint32_t));
        }
        return 0;
    }
}

int vxt_metal_turbo_rollback_deltanet(
    void* raw_runner,
    void* raw_state,
    uint32_t accepted_m,
    uint32_t batch_m,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* r = static_cast<MetalTurboRunner*>(raw_runner);
        auto* s = static_cast<MetalTurboState*>(raw_state);
        if (!r || !s || accepted_m == 0u || accepted_m > MAX_BATCH_M || batch_m == 0u || batch_m > MAX_BATCH_M) {
            vxt_write_err(err_out, err_cap, "invalid args to vxt_metal_turbo_rollback_deltanet");
            return -1;
        }
        if (accepted_m >= batch_m) return 0;

        const VxtLlmConfig& c = r->cfg;
        bool any_rec = false;
        for (uint8_t ir : s->is_rec) if (ir) { any_rec = true; break; }
        if (!any_rec) return 0;

        id<MTLCommandBuffer> cmd = [r->gemv.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        for (uint32_t il = 0; il < c.n_layer; ++il) {
            if (!s->is_rec[il]) continue;
            if (s->conv_bytes[il] > 0) {
                [blit copyFromBuffer:s->conv_bak[il] sourceOffset:0
                            toBuffer:s->conv_states[il] destinationOffset:0
                                size:s->conv_bytes[il]];
            }
            if (s->ssm_bytes[il] > 0) {
                [blit copyFromBuffer:s->ssm_bak[il] sourceOffset:0
                            toBuffer:s->ssm_states[il] destinationOffset:0
                                size:s->ssm_bytes[il]];
            }
        }
        [blit endEncoding];

        uint32_t replay_m = accepted_m - 1u;
        if (replay_m > 0u) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            uint32_t n_kh = c.ssm_group_count;
            uint32_t n_vh = c.ssm_dt_rank;
            uint32_t dk = c.ssm_state_size;
            uint32_t dv = c.ssm_inner_size / n_vh;
            uint32_t conv_ch = 2u * n_kh * dk + c.ssm_inner_size;
            uint32_t padded_inner = ((c.ssm_inner_size + 255u) / 256u) * 256u;
            uint32_t padded_embd = ((c.n_embd + 255u) / 256u) * 256u;

            for (uint32_t il = 0; il < c.n_layer; ++il) {
                if (!s->is_rec[il]) continue;
                const auto& L = r->layers[il];
                size_t qkv_m1_off = size_t(L.w_qkv.rows) * sizeof(float);
                TurboDeltaConvParamsC cp{ conv_ch, c.ssm_conv_kernel, L.w_qkv.rows, replay_m, 0u };
                [enc setComputePipelineState:r->pso_deltanet_conv_batch];
                [enc setBuffer:s->qkv_hist[il] offset:qkv_m1_off atIndex:0];
                [enc setBuffer:r->weight_arena_buf offset:L.vec0_off atIndex:1];
                [enc setBuffer:s->conv_states[il] offset:0 atIndex:2];
                [enc setBuffer:s->conv_bak[il] offset:0 atIndex:3];
                [enc setBuffer:r->conv_silu_buf offset:0 atIndex:4];
                [enc setBytes:&cp length:sizeof(cp) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake((conv_ch + 255u) / 256u, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                id<MTLBuffer> z_buf = (L.has_split_gate != 0u) ? s->z_sep_hist[il] : s->qkv_hist[il];
                uint32_t z_stride = (L.has_split_gate != 0u) ? c.ssm_inner_size : L.w_qkv.rows;
                size_t z_off = (L.has_split_gate != 0u)
                    ? size_t(c.ssm_inner_size) * sizeof(float)
                    : size_t(L.w_qkv.rows + conv_ch) * sizeof(float);
                size_t ab_m1_off = size_t(2u * n_vh) * sizeof(float);

                TurboDeltaHeadsParamsC hp{
                    n_kh, n_vh, dk, dv, L.w_qkv.rows, z_stride, padded_inner, padded_embd,
                    replay_m, 0u, c.rms_norm_eps, 0u, 0u
                };
                [enc setComputePipelineState:r->pso_deltanet_heads_fp16];
                [enc setBuffer:r->conv_silu_buf offset:0 atIndex:0];
                [enc setBuffer:z_buf offset:z_off atIndex:1];
                [enc setBuffer:s->ab_hist[il] offset:ab_m1_off atIndex:2];
                [enc setBuffer:r->weight_arena_buf offset:L.vec1_off atIndex:3];
                [enc setBuffer:r->weight_arena_buf offset:L.vec2_off atIndex:4];
                [enc setBuffer:r->weight_arena_buf offset:L.vec3_off atIndex:5];
                [enc setBuffer:s->ssm_states[il] offset:0 atIndex:6];
                [enc setBuffer:s->ssm_bak[il] offset:0 atIndex:7];
                [enc setBuffer:r->x_rot_buf offset:0 atIndex:8];
                [enc setBytes:&hp length:sizeof(hp) atIndex:9];
                [enc setBuffer:r->x_rot_h_buf offset:0 atIndex:10];
                [enc dispatchThreadgroups:MTLSizeMake(n_vh, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 0 : -2;
    }
}

// Generation owns the existing backup buffers until all queued M=1 steps drain.
// Those steps use save_backup=0, so they never overwrite this checkpoint.
int vxt_metal_turbo_generation_checkpoint(void* raw_runner, void* raw_state,
                                         uint32_t restore, char* err_out, size_t err_cap) {
    @autoreleasepool {
        auto* r = static_cast<MetalTurboRunner*>(raw_runner);
        auto* s = static_cast<MetalTurboState*>(raw_state);
        if (!r || !s) {
            vxt_write_err(err_out, err_cap, "null Turbo generation checkpoint");
            return -1;
        }
        id<MTLCommandBuffer> cmd = [r->gemv.queue commandBuffer];
        id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
        if (!cmd || !enc) {
            vxt_write_err(err_out, err_cap, "cannot create Turbo checkpoint command");
            return -1;
        }
        for (uint32_t il = 0; il < s->n_layer; ++il) {
            if (!s->is_rec[il]) continue;
            auto copy = [&](id<MTLBuffer> live, id<MTLBuffer> backup, size_t bytes) {
                if (bytes) [enc copyFromBuffer:(restore ? backup : live) sourceOffset:0
                                     toBuffer:(restore ? live : backup) destinationOffset:0 size:bytes];
            };
            copy(s->conv_states[il], s->conv_bak[il], s->conv_bytes[il]);
            copy(s->ssm_states[il], s->ssm_bak[il], s->ssm_bytes[il]);
        }
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            vxt_write_err(err_out, err_cap, "Turbo generation checkpoint failed");
            return -1;
        }
        return 0;
    }
}

void vxt_metal_turbo_ring_set_token(void* raw_runner, uint32_t slot, uint32_t tok_id) {
    auto* r = static_cast<MetalTurboRunner*>(raw_runner);
    if (!r) return;
    static_cast<uint32_t*>([r->tok_ring_buf contents])[slot & (RING_SLOTS - 1u)] = tok_id;
}

uint32_t vxt_metal_turbo_ring_get_out(void* raw_runner, uint32_t slot) {
    auto* r = static_cast<MetalTurboRunner*>(raw_runner);
    if (!r) return 0u;
    return static_cast<const uint32_t*>([r->out_ring_buf contents])[slot & (RING_SLOTS - 1u)];
}

void* vxt_metal_turbo_step_async_enqueue(
    void* raw_runner,
    void* raw_state,
    uint32_t base_slot,
    uint32_t step_count,
    uint32_t pos_base,
    uint32_t use_sketch,
    char* err_out,
    size_t err_cap
) {
    @autoreleasepool {
        auto* r = static_cast<MetalTurboRunner*>(raw_runner);
        auto* s = static_cast<MetalTurboState*>(raw_state);
        if (!r || !s || step_count == 0u) {
            vxt_write_err(err_out, err_cap, "null runner/state in step_async_enqueue");
            return nullptr;
        }
        // Growth copies shared cache memory on the CPU. Earlier pipelined
        // commands may still be writing the old allocation: finish that queue
        // prefix before copying/replacing it. No fence on the normal fast path.
        if (s->kv_stride != 0u && pos_base + step_count + 200u >= s->cap_seq) {
            id<MTLCommandBuffer> fence = [r->gemv.queue commandBuffer];
            if (!fence) {
                vxt_write_err(err_out, err_cap, "cannot create Turbo KV growth fence");
                return nullptr;
            }
            [fence commit];
            [fence waitUntilCompleted];
            if ([fence status] != MTLCommandBufferStatusCompleted) {
                vxt_write_err(err_out, err_cap, "Turbo KV growth fence failed");
                return nullptr;
            }
        }
        if (!s->ensure_kv_cap(pos_base + step_count + 200u)) {
            vxt_write_err(err_out, err_cap, "failed to grow Turbo KV cache in async enqueue");
            return nullptr;
        }
        bool use_conc_wait = false;
        id<MTLCommandBuffer> cmd = [r->gemv.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = use_conc_wait
            ? [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
            : [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            vxt_write_err(err_out, err_cap, "failed to create async Metal command buffer");
            return nullptr;
        }
        uint32_t mode = (use_sketch != 0u) ? 2u : 3u;

        for (uint32_t k = 0u; k < step_count; ++k) {
            uint32_t slot = (base_slot + k) & (RING_SLOTS - 1u);
            uint32_t next_slot = (base_slot + k + 1u) & (RING_SLOTS - 1u);
            uint32_t pos_k = pos_base + k;
            size_t slot_off = size_t(slot) * sizeof(uint32_t);
            encode_turbo_forward(
                r, s, enc, r->tok_ring_buf, slot_off, 1u, pos_k, 0u,
                mode, r->out_ring_buf, slot_off, next_slot,
                0xffffffffu, 0xffffffffu, false, -1, use_conc_wait
            );
        }
        [enc endEncoding];
        [cmd commit];
        return (__bridge_retained void*)cmd;
    }
}

int vxt_metal_turbo_wait_cmd(void* raw_cmd, char* err_out, size_t err_cap) {
    @autoreleasepool {
        if (!raw_cmd) return 0;
        id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)raw_cmd;
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            NSString* desc = [[cmd error] localizedDescription];
            vxt_write_err(err_out, err_cap, desc ? [desc UTF8String] : "async Metal command buffer failed");
            return -1;
        }
        return 0;
    }
}

} // extern "C"
