// ============================================================================
// vxt_llm_turbo.metal — Ultra-Fused Zero-SRAM Register-Shuffle `Turbo` LLM Kernels
// (`Embed`, `Fused-FWHT32 RMSNorm`, `Fused Gate+Up+SwiGLU+FWHT32`, `Fused 1-to-1
//  Conv1D+DeltaNet+FWHT32`, `Zero-SRAM Register-Shuffle GEMV/GEMM`, `1024-Bit
//  Hadamard Sign-Sketch + Quantile Top-128 Block-Exact LM Head`, `All-GPU Argmax`)
// ============================================================================

#if defined(__HAVE_COHERENT__)
#define VXT_COHERENT_DEV device coherent(device)
#else
#define VXT_COHERENT_DEV device
#endif

inline void vxt_device_fence() {
#if defined(__HAVE_ATOMIC_FENCE__)
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#else
    threadgroup_barrier(mem_flags::mem_device);
#endif
}

struct TurboEmbedParams {
    uint cols;
    uint padded_cols;
    uint bits;
    uint batch_m;
};

struct TurboEmbedNormParams {
    uint cols;
    uint padded_cols;
    uint bits;
    float eps;
};

struct TurboNormParams {
    uint dim;
    uint padded_cols;
    float eps;
    uint batch_m;
};

struct TurboSwigluParams {
    uint n_ff;
    uint padded_cols;
    uint gu_stride;
    uint batch_m;
};

struct TurboLlmMatParams {
    uint rows;
    uint padded_cols;
    uint cols;
    uint flags;
    uint norm_dim;
    float norm_eps;
    uint next_padded_cols;
    uint sync_slot;
    uint num_tgs;
    uint _pad;
};

struct TurboFfnFusedParams {
    uint n_ff;
    uint padded_ff;
    uint padded_cols;
    uint bits;
    uint k_splits;
    uint sync_base;
    uint norm_dim;
    float norm_eps;
    uint fuse_norm;
    uint _pad;
};

struct TurboTailNormParams {
    uint rows;
    uint padded_cols;
    uint next_padded_cols;
    uint flags;
    uint tile_sync_base;
    uint mat_sync_slot;
    float eps;
    uint _pad;
};

struct TurboRaw16Params {
    uint rows;
    uint padded_cols;
    uint batch_m;
    uint _pad;
};

struct TurboGemmParams {
    uint rows;
    uint padded_cols;
    uint batch_m;
    uint tiles_per_sg;
};

struct TurboDeltaConvParams {
    uint conv_ch;
    uint k_sz;
    uint qkv_gate_stride;
    uint batch_m;
    uint save_backup;
};

struct TurboDeltaHeadsParams {
    uint n_kh;
    uint n_vh;
    uint dk;
    uint dv;
    uint qkv_stride;
    uint z_stride;
    uint padded_inner;
    uint padded_embd;
    uint batch_m;
    uint save_backup;
    float eps;
    uint fuse_out;
    uint sync_slot;
};

struct TurboGqaParams {
    uint nh;
    uint nkv;
    uint hd;
    uint pos_base;
    uint rope_dim;
    uint flags;
    uint qkv_stride;
    uint kv_stride;
    uint q_raw_dim;
    uint padded_qdim;
    uint batch_m;
    float rope_freq_base;
    float eps;
};

struct TurboProjMegaParams {
    uint padded_cols;
    uint rows0;
    uint bits0;
    uint groups0;
    uint rows1;
    uint bits1;
    uint groups1;
    uint ab_rows;
};

struct TurboArgmaxParams {
    uint n_vocab;
    uint n_blocks;
    uint batch_m;
    uint write_next_slot;
    float presence_pen;
    uint d1;
    uint d2;
    uint d3;
};

struct TurboSketchBuildParams {
    uint rows;
    uint padded_cols;
    uint bits;
    uint stride_tiles8;
};

struct TurboSketchEvalParams {
    uint rows;
    uint padded_cols;
    uint n_blocks;
    uint stride_tiles8;
    float presence_pen;
    uint d1;
    uint d2;
    uint d3;
};

struct TurboSketchStage2Params {
    uint rows;
    uint padded_cols;
    uint bits;
    float presence_pen;
    uint stage2_tgs;
    uint d1;
    uint d2;
    uint d3;
};

// Dual-Pipe Threadgroup-SRAM + Constant-L1 FP16 VQ Decoders (`0` simd_shuffle, 2x FP16 FMA throughput!)
inline uint vxt_t3_unpack_code6(uint rw0, uint rw1, uint rw2, uint sr, uint pr) {
    if (sr == 0u) return (rw0 >> (pr * 6u)) & 63u;
    if (sr == 1u) return (rw1 >> (pr * 6u)) & 63u;
    if (sr == 2u) return (rw2 >> (pr * 6u)) & 63u;
    if (pr == 0u) return (rw0 >> 24u) & 63u;
    if (pr == 1u) return (rw1 >> 24u) & 63u;
    if (pr == 2u) return (rw2 >> 24u) & 63u;
    return (rw0 >> 30u) | ((rw1 >> 28u) & 0x0Cu) | ((rw2 >> 26u) & 0x30u);
}

inline half vxt_dot2_h4(half4 v0, half4 v1, half4 xh0, half4 xh1) {
    half2 p = v0.xy * xh0.xy;
    p = fma(v0.zw, xh0.zw, p);
    p = fma(v1.xy, xh1.xy, p);
    p = fma(v1.zw, xh1.zw, p);
    return p.x + p.y;
}

inline float4 vxt_eval_t2_lut(
    threadgroup const half4* lut256,
    uint2 w,
    half4 sh,
    half4 xh0,
    half4 xh1,
    float4 acc
) {
    uchar4 b01 = as_type<uchar4>(w.x);
    uchar4 b23 = as_type<uchar4>(w.y);
    half d0 = vxt_dot2_h4(lut256[b01.x], lut256[b01.y], xh0, xh1);
    half d1 = vxt_dot2_h4(lut256[b01.z], lut256[b01.w], xh0, xh1);
    half d2 = vxt_dot2_h4(lut256[b23.x], lut256[b23.y], xh0, xh1);
    half d3 = vxt_dot2_h4(lut256[b23.z], lut256[b23.w], xh0, xh1);
    return fma(float4(sh), float4(half4(d0, d1, d2, d3)), acc);
}

inline half vxt_dot_t3_row24(threadgroup const half2* lut64, uint rw, half4 xh0, half4 xh1) {
    half2 p = lut64[rw & 63u] * xh0.xy;
    p = fma(lut64[(rw >> 6u) & 63u], xh0.zw, p);
    p = fma(lut64[(rw >> 12u) & 63u], xh1.xy, p);
    p = fma(lut64[(rw >> 18u) & 63u], xh1.zw, p);
    return p.x + p.y;
}

inline float4 vxt_eval_t3_lut(
    threadgroup const half2* lut64,
    uint2 rw01,
    uint rw2,
    half4 sh,
    half4 xh0,
    half4 xh1,
    float4 acc
) {
    half d0 = vxt_dot_t3_row24(lut64, rw01.x, xh0, xh1);
    half d1 = vxt_dot_t3_row24(lut64, rw01.y, xh0, xh1);
    half d2 = vxt_dot_t3_row24(lut64, rw2, xh0, xh1);
    uint c33 = (rw01.x >> 30u) | ((rw01.y >> 28u) & 0x0Cu) | ((rw2 >> 26u) & 0x30u);
    half2 p3 = lut64[(rw01.x >> 24u) & 63u] * xh0.xy;
    p3 = fma(lut64[(rw01.y >> 24u) & 63u], xh0.zw, p3);
    p3 = fma(lut64[(rw2 >> 24u) & 63u], xh1.xy, p3);
    p3 = fma(lut64[c33], xh1.zw, p3);
    half d3 = p3.x + p3.y;
    return fma(float4(sh), float4(half4(d0, d1, d2, d3)), acc);
}

inline float4 vxt_eval_t25_lut(
    threadgroup const half2* lut_lo,
    threadgroup const half2* lut_hi,
    uint2 w01,
    uint h16,
    half4 sh,
    half4 xh0,
    half4 xh1,
    float4 acc
) {
    uint c00 = w01.x & 1023u, c01 = (w01.x >> 10u) & 1023u;
    uint c10 = (w01.x >> 20u) & 1023u, c11 = w01.y & 1023u;
    uint c20 = (w01.y >> 10u) & 1023u, c21 = (w01.y >> 20u) & 1023u;
    uint c30 = (h16 & 0xffu) | ((w01.x >> 30u) << 8u);
    uint c31 = (h16 >> 8u) | ((w01.y >> 30u) << 8u);
    half d0 = vxt_dot2_h4(vxt_t25_vec_swz(lut_lo, lut_hi, c00), vxt_t25_vec_swz(lut_lo, lut_hi, c01), xh0, xh1);
    half d1 = vxt_dot2_h4(vxt_t25_vec_swz(lut_lo, lut_hi, c10), vxt_t25_vec_swz(lut_lo, lut_hi, c11), xh0, xh1);
    half d2 = vxt_dot2_h4(vxt_t25_vec_swz(lut_lo, lut_hi, c20), vxt_t25_vec_swz(lut_lo, lut_hi, c21), xh0, xh1);
    half d3 = vxt_dot2_h4(vxt_t25_vec_swz(lut_lo, lut_hi, c30), vxt_t25_vec_swz(lut_lo, lut_hi, c31), xh0, xh1);
    return fma(float4(sh), float4(half4(d0, d1, d2, d3)), acc);
}

inline float4 vxt_eval_t25_lut1k(
    threadgroup const half4* lut1k,
    uint2 w01,
    uint h16,
    half4 sh,
    half4 xh0,
    half4 xh1,
    float4 acc
) {
    half d0 = vxt_dot2_h4(lut1k[w01.x & 1023u], lut1k[(w01.x >> 10u) & 1023u], xh0, xh1);
    half d1 = vxt_dot2_h4(lut1k[(w01.x >> 20u) & 1023u], lut1k[w01.y & 1023u], xh0, xh1);
    half d2 = vxt_dot2_h4(lut1k[(w01.y >> 10u) & 1023u], lut1k[(w01.y >> 20u) & 1023u], xh0, xh1);
    half d3 = vxt_dot2_h4(lut1k[(h16 & 0xffu) | ((w01.x >> 30u) << 8u)], lut1k[(h16 >> 8u) | ((w01.y >> 30u) << 8u)], xh0, xh1);
    return fma(float4(sh), float4(half4(d0, d1, d2, d3)), acc);
}

inline void vxt_accum_tile4_coh(VXT_COHERENT_DEV float* y, uint rows, uint tile, float4 sum4, bool add_res) {
    uint r = tile << 2u;
    if (!add_res) {
        if (r + 3u < rows) {
            *(VXT_COHERENT_DEV float4*)(y + r) = sum4;
        } else {
            if (r < rows) y[r] = sum4.x;
            if (r + 1u < rows) y[r + 1u] = sum4.y;
            if (r + 2u < rows) y[r + 2u] = sum4.z;
        }
        return;
    }
    if (r + 3u < rows) {
        *(VXT_COHERENT_DEV float4*)(y + r) += sum4;
    } else {
        if (r < rows) y[r] += sum4.x;
        if (r + 1u < rows) y[r + 1u] += sum4.y;
        if (r + 2u < rows) y[r + 2u] += sum4.z;
    }
}

kernel void vxt_build_lut_bank(
    device half4* lut_bank  [[buffer(0)]],
    device uint* caps_out   [[buffer(1)]],
    uint tid                [[thread_index_in_threadgroup]],
    uint3 tgp               [[threadgroup_position_in_grid]]
) {
    uint idx = tgp.x * 128u + tid;
    if (idx < 1024u) {
        lut_bank[idx] = vxt_orth4d64_const(vxt_t25_unswz(idx));
    }
    if (idx < 256u) {
        ushort4 u = ORTH4D_16_U16[idx & 15u];
        u.x |= ushort((idx & 0x10u) << 11u);
        u.y |= ushort((idx & 0x20u) << 10u);
        u.z |= ushort((idx & 0x40u) << 9u);
        u.w |= ushort((idx & 0x80u) << 8u);
        lut_bank[1024u + idx] = as_type<half4>(u);
    }
    if (idx < 64u) {
        uint idx4 = idx & 15u;
        ushort2 bt = ORTH2D_16_U16[idx4];
        ushort2 be = EC_ORTH2D_16_U16[idx4];
        ushort s0 = ushort((idx << 11u) & 0x8000u);
        ushort s1 = ushort((idx << 10u) & 0x8000u);
        ((device half2*)(lut_bank + 1280u))[idx] = as_type<half2>(ushort2(
            bt.x == 0u ? 0u : (bt.x | s0), bt.y == 0u ? 0u : (bt.y | s1)
        ));
        ((device half2*)(lut_bank + 1312u))[idx] = as_type<half2>(ushort2(
            be.x == 0u ? 0u : (be.x | s0), be.y == 0u ? 0u : (be.y | s1)
        ));
    }
    if (idx == 0u) {
#if defined(__HAVE_COHERENT__) && defined(__HAVE_ATOMIC_FENCE__)
        caps_out[0] = 1u;
#else
        caps_out[0] = 0u;
#endif
    }
}

// 1. All-GPU Compressed Embedding Lookup + Inverse FWHT32 + Column Sign Flip (M = 1..8)
kernel void vxt_embed_turbo(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const uint* tok_ids      [[buffer(2)]],
    device float* cur_out           [[buffer(3)]],
    constant TurboEmbedParams& p    [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint step = tg_pos.x;
    uint m = tg_pos.y;
    uint steps = p.padded_cols >> 8u;
    if (step >= steps || m >= p.batch_m) return;

    uint r = tok_ids[m];
    uint tile = r >> 2u;
    uint sr = r & 3u;
    uint gh = tid >> 7u;
    uint lane_w = (tid & 255u) >> 3u;
    uint k = tid & 7u;

    float sc = float(as_type<half>(scales_f16[(ulong(tile) * ulong(steps) + ulong(step)) * 8ul + ulong(gh * 4u + sr)]));
    float rbuf_elem = 0.0f;

    if (p.bits == 2u) {
        device const uint* p64 = packed_words + (ulong(tile) * ulong(steps) + ulong(step)) * 64ul;
        uint w = p64[lane_w * 2u + (sr >> 1u)];
        uint c16 = (w >> ((sr & 1u) * 16u)) & 0xffffu;
        uint byte_b = (k < 4u) ? (c16 & 0xffu) : (c16 >> 8u);
        float4 v4 = float4(vxt_orth4d_const(byte_b, ORTH4D_16_U16));
        rbuf_elem = sc * v4[k & 3u];
    } else if (p.bits == 5u) {
        device const uint* p80 = packed_words + (ulong(tile) * ulong(steps) + ulong(step)) * 80ul;
        uint2 cc = vxt_t25_unpack_row(p80, lane_w, sr);
        float4 v4 = float4(vxt_orth4d64_const((k < 4u) ? cc.x : cc.y));
        rbuf_elem = sc * v4[k & 3u];
    } else {
        device const uint* p96 = packed_words + (ulong(tile) * ulong(steps) + ulong(step)) * 96ul;
        uint code6 = vxt_t3_unpack_code6(p96[lane_w * 2u], p96[lane_w * 2u + 1u], p96[64u + lane_w], sr, k >> 1u);
        float2 v2 = float2(vxt_orth2d_const_sel(code6, p.bits == 4u));
        rbuf_elem = sc * v2[k & 1u];
    }

    float unrot = vxt_fwht32_lane(rbuf_elem, lane);
    uint col = (step << 8u) + tid;
    if (col < p.cols) {
        cur_out[ulong(m) * ulong(p.cols) + ulong(col)] = vxt_apply_col_sign(unrot, col);
    }
}

// 1b. Single-TG Fused Embedding Decode + Layer-0 RMSNorm + Sign-Flip + FWHT32 (`M = 1`, `padded_cols <= 4096`)
kernel void vxt_embed_norm0_m1(
    device const ushort* scales_f16   [[buffer(0)]],
    device const uint* packed_words   [[buffer(1)]],
    device const uint* tok_ids        [[buffer(2)]],
    VXT_COHERENT_DEV float* cur_out   [[buffer(3)]],
    device const float* norm0_w       [[buffer(4)]],
    device float* x_rot_out           [[buffer(5)]],
    constant TurboEmbedNormParams& p  [[buffer(6)]],
    device half* x_rot_h_out          [[buffer(7)]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint sg_id                        [[simdgroup_index_in_threadgroup]],
    uint lane                         [[thread_index_in_simdgroup]]
) {
    uint r = tok_ids[0];
    uint tile = r >> 2u;
    uint sr = r & 3u;
    uint steps = p.padded_cols >> 8u;
    uint gh = tid >> 7u;
    uint lane_w = (tid & 255u) >> 3u;
    uint k = tid & 7u;

    float reg_v[16];
    float sq = 0.0f;
    for (uint step = 0u; step < steps; ++step) {
        float sc = float(as_type<half>(scales_f16[(ulong(tile) * ulong(steps) + ulong(step)) * 8ul + ulong(gh * 4u + sr)]));
        float rbuf_elem = 0.0f;
        if (p.bits == 2u) {
            device const uint* p64 = packed_words + (ulong(tile) * ulong(steps) + ulong(step)) * 64ul;
            uint w = p64[lane_w * 2u + (sr >> 1u)];
            uint c16 = (w >> ((sr & 1u) * 16u)) & 0xffffu;
            uint byte_b = (k < 4u) ? (c16 & 0xffu) : (c16 >> 8u);
            float4 v4 = float4(vxt_orth4d_const(byte_b, ORTH4D_16_U16));
            rbuf_elem = sc * v4[k & 3u];
        } else if (p.bits == 5u) {
            device const uint* p80 = packed_words + (ulong(tile) * ulong(steps) + ulong(step)) * 80ul;
            uint2 cc = vxt_t25_unpack_row(p80, lane_w, sr);
            float4 v4 = float4(vxt_orth4d64_const((k < 4u) ? cc.x : cc.y));
            rbuf_elem = sc * v4[k & 3u];
        } else {
            device const uint* p96 = packed_words + (ulong(tile) * ulong(steps) + ulong(step)) * 96ul;
            uint code6 = vxt_t3_unpack_code6(p96[lane_w * 2u], p96[lane_w * 2u + 1u], p96[64u + lane_w], sr, k >> 1u);
            float2 v2 = float2(vxt_orth2d_const_sel(code6, p.bits == 4u));
            rbuf_elem = sc * v2[k & 1u];
        }

        float unrot = vxt_fwht32_lane(rbuf_elem, lane);
        uint col = (step << 8u) + tid;
        float v = 0.0f;
        if (col < p.cols) {
            v = vxt_apply_col_sign(unrot, col);
            cur_out[col] = v;
            sq = fma(v, v, sq);
        }
        reg_v[step] = v;
    }

    sq = simd_sum(sq);
    threadgroup float sg_sums[8];
    if (lane == 0u) sg_sums[sg_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float v = (lane < 8u) ? sg_sums[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) sg_sums[0] = rsqrt(v / float(p.cols) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = sg_sums[0];

    for (uint step = 0u; step < steps; ++step) {
        uint col = (step << 8u) + tid;
        float val = (col < p.cols) ? vxt_apply_col_sign(reg_v[step] * inv_rms * norm0_w[col], col) : 0.0f;
        float rot = vxt_fwht32_lane(val, lane);
        x_rot_out[col] = rot;
        x_rot_h_out[col] = half(rot * 0.0625f);
    }
}

// 2. Fused RMSNorm + Sign-Flip + FWHT32 -> x_rot_out (M = 1..8)
kernel void vxt_rms_norm_turbo(
    device const float* x       [[buffer(0)]],
    device const float* w       [[buffer(1)]],
    device float* x_rot_out     [[buffer(2)]],
    constant TurboNormParams& p [[buffer(3)]],
    device half* x_rot_h_out    [[buffer(4)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint3 tg_pos                [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    uint batch_m = p.batch_m & 255u;
    bool skip_f32 = (p.batch_m & 256u) != 0u;
    if (m >= batch_m) return;
    device const float* xm = x + ulong(m) * ulong(p.dim);
    device float* rot_m = x_rot_out + ulong(m) * ulong(p.padded_cols);

    float sq = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        float v = xm[i];
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    threadgroup float sg_sums[8];
    if (lane == 0u) sg_sums[sg_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float v = (lane < 8u) ? sg_sums[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) sg_sums[0] = rsqrt(v / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = sg_sums[0];

    uint n_blks = p.padded_cols >> 5u;
    device half* rot_hm = x_rot_h_out + ulong(m) * ulong(p.padded_cols);
    for (uint blk = sg_id; blk < n_blks; blk += 8u) {
        uint col = (blk << 5u) + lane;
        float val = (col < p.dim) ? vxt_apply_col_sign(xm[col] * scale * w[col], col) : 0.0f;
        float rot = vxt_fwht32_lane(val, lane);
        if (!skip_f32) rot_m[col] = rot;
        rot_hm[col] = half(rot * 0.0625f);
    }
}

// 3. Fused Residual-Add + RMSNorm + Sign-Flip + FWHT32 -> x_rot_out (M = 1..8)
kernel void vxt_add_rms_norm_turbo(
    device float* cur           [[buffer(0)]],
    device const float* delta   [[buffer(1)]],
    device const float* w       [[buffer(2)]],
    device float* x_rot_out     [[buffer(3)]],
    constant TurboNormParams& p [[buffer(4)]],
    device half* x_rot_h_out    [[buffer(5)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint3 tg_pos                [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    if (m >= p.batch_m) return;
    device float* cm = cur + ulong(m) * ulong(p.dim);
    device const float* dm = delta + ulong(m) * ulong(p.dim);
    device float* rot_m = x_rot_out + ulong(m) * ulong(p.padded_cols);

    float sq = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        float v = cm[i] + dm[i];
        cm[i] = v;
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    threadgroup float sg_sums[8];
    if (lane == 0u) sg_sums[sg_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float v = (lane < 8u) ? sg_sums[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) sg_sums[0] = rsqrt(v / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = sg_sums[0];

    uint n_blks = p.padded_cols >> 5u;
    device half* rot_hm = x_rot_h_out + ulong(m) * ulong(p.padded_cols);
    for (uint blk = sg_id; blk < n_blks; blk += 8u) {
        uint col = (blk << 5u) + lane;
        float val = (col < p.dim) ? vxt_apply_col_sign(cm[col] * scale * w[col], col) : 0.0f;
        float rot = vxt_fwht32_lane(val, lane);
        rot_m[col] = rot;
        rot_hm[col] = half(rot * 0.0625f);
    }
}

// 4. Fused SwiGLU (`gate_up` contiguous) + Sign-Flip + FWHT32 -> x_rot_out (used for M > 1)
kernel void vxt_swiglu_turbo(
    device const float* gate_up   [[buffer(0)]],
    device float* x_rot_out       [[buffer(1)]],
    constant TurboSwigluParams& p [[buffer(2)]],
    device half* x_rot_h_out      [[buffer(3)]],
    uint sg_id                    [[simdgroup_index_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]],
    uint3 tg_pos                  [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    uint blk = tg_pos.x * 8u + sg_id;
    uint col = (blk << 5u) + lane;
    if (m >= p.batch_m || col >= p.padded_cols) return;

    float val = 0.0f;
    if (col < p.n_ff) {
        device const float* gu_m = gate_up + ulong(m) * ulong(p.gu_stride);
        float g = ((p.n_ff & 31u) == 0u) ? gu_m[(blk << 6u) + lane] : gu_m[col];
        float u = ((p.n_ff & 31u) == 0u) ? gu_m[(blk << 6u) + 32u + lane] : gu_m[p.n_ff + col];
        float sw = (g / (1.0f + exp(-g))) * u;
        val = vxt_apply_col_sign(sw, col);
    }
    float rot = vxt_fwht32_lane(val, lane);
    ulong out_idx = ulong(m) * ulong(p.padded_cols) + ulong(col);
    if (p.batch_m > 1u) x_rot_out[out_idx] = rot;
    x_rot_h_out[out_idx] = half(rot * 0.0625f);
}

// 5. Pre-Rotated FP16 `[ssm_alpha; ssm_beta]` GEMV/GEMM reading `x_rot` directly (M = 1..4)
kernel void vxt_llm_raw16_rot(
    device const half* w_rot_f16 [[buffer(0)]],
    device const float* x_rot    [[buffer(1)]],
    device float* y_out          [[buffer(2)]],
    constant TurboRaw16Params& p [[buffer(3)]],
    uint sg_id                   [[simdgroup_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 2u + sg_id;
    if (row >= p.rows) return;

    device const half4* wr4 = (device const half4*)(w_rot_f16 + ulong(row) * ulong(p.padded_cols));
    uint n4 = p.padded_cols >> 2u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (uint c4 = lane; c4 < n4; c4 += 32u) {
        float4 wv = float4(wr4[c4]);
        acc0 += dot(wv, *(device const float4*)(x_rot + (c4 << 2u)));
        if (p.batch_m > 1u) acc1 += dot(wv, *(device const float4*)(x_rot + p.padded_cols + (c4 << 2u)));
        if (p.batch_m > 2u) acc2 += dot(wv, *(device const float4*)(x_rot + 2u * p.padded_cols + (c4 << 2u)));
        if (p.batch_m > 3u) acc3 += dot(wv, *(device const float4*)(x_rot + 3u * p.padded_cols + (c4 << 2u)));
    }

    float s0 = simd_sum(acc0);
    float s1 = (p.batch_m > 1u) ? simd_sum(acc1) : 0.0f;
    float s2 = (p.batch_m > 2u) ? simd_sum(acc2) : 0.0f;
    float s3 = (p.batch_m > 3u) ? simd_sum(acc3) : 0.0f;
    if (lane == 0u) {
        y_out[row] = s0;
        if (p.batch_m > 1u) y_out[p.rows + row] = s1;
        if (p.batch_m > 2u) y_out[2u * p.rows + row] = s2;
        if (p.batch_m > 3u) y_out[3u * p.rows + row] = s3;
    }
}

// 5b. Lean 100%-Occupancy `M = 1` GEMV Kernels (`1sg` 256t, `sk2` 128t/256t/512t + Zero-Spin Last-TG Epilogue Norm, & 512t Split-K `ffn_gu+SwiGLU`)
inline void vxt_epilogue_last_tg_norm(
    VXT_COHERENT_DEV const float* y,
    device const float* norm_w,
    device half* x_rot_h_out,
    device float* x_rot_out,
    device atomic_uint* sync_cnt,
    threadgroup float* sg_s,
    constant TurboLlmMatParams& p,
    uint tg_threads,
    uint tid,
    uint sg_id,
    uint lane
) {
    if ((p.flags & (1024u | 4096u)) == 0u) return;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0u) {
        vxt_device_fence();
        uint prev = atomic_fetch_add_explicit(&sync_cnt[p.sync_slot], 1u, memory_order_relaxed);
        if (prev + 1u == p.num_tgs) {
            atomic_store_explicit(&sync_cnt[p.sync_slot], 0u, memory_order_relaxed);
            sg_s[16] = 1.0f;
        } else {
            sg_s[16] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_s[16] == 0.0f) return;

    vxt_device_fence();
    uint n_sgs = tg_threads >> 5u;
    uint n_blks = p.next_padded_cols >> 5u;
    if ((p.flags & 4096u) != 0u) {
        uint n_ff = p.cols;
        for (uint blk = sg_id; blk < n_blks; blk += n_sgs) {
            uint col = (blk << 5u) + lane;
            float val = 0.0f;
            if (col < n_ff) {
                float g = ((n_ff & 31u) == 0u) ? y[(blk << 6u) + lane] : y[col];
                float u = ((n_ff & 31u) == 0u) ? y[(blk << 6u) + 32u + lane] : y[n_ff + col];
                val = vxt_apply_col_sign((g / (1.0f + exp(-g))) * u, col);
            }
            float rot = vxt_fwht32_lane(val, lane);
            x_rot_h_out[col] = half(rot * 0.0625f);
        }
        return;
    }

    float sq = 0.0f;
    for (uint i = tid; i < p.norm_dim; i += tg_threads) {
        float v = y[i];
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u) sg_s[sg_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float v = (lane < n_sgs) ? sg_s[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) sg_s[0] = rsqrt(v / float(p.norm_dim) + p.norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = sg_s[0];
    bool write_f = (p.flags & 2048u) != 0u;
    for (uint blk = sg_id; blk < n_blks; blk += n_sgs) {
        uint col = (blk << 5u) + lane;
        float val = (col < p.norm_dim) ? vxt_apply_col_sign(y[col] * inv_rms * norm_w[col], col) : 0.0f;
        float rot = vxt_fwht32_lane(val, lane);
        x_rot_h_out[col] = half(rot * 0.0625f);
        if (write_f) x_rot_out[col] = rot;
    }
}

inline void vxt_signal_tg_done(device atomic_uint* sync_cnt, uint flags, uint sync_slot, uint tid) {
    (void)sync_cnt; (void)flags; (void)sync_slot; (void)tid;
}

kernel void vxt_rms_norm_wait_m1(
    VXT_COHERENT_DEV const float* cur_in [[buffer(0)]],
    device const float* norm_w           [[buffer(1)]],
    device float* x_rot_out              [[buffer(2)]],
    constant TurboTailNormParams& p      [[buffer(3)]],
    device half* x_rot_h_out             [[buffer(4)]],
    device atomic_uint* sync_cnt         [[buffer(5)]],
    uint tid                             [[thread_index_in_threadgroup]],
    uint sg_id                           [[simdgroup_index_in_threadgroup]],
    uint lane                            [[thread_index_in_simdgroup]]
) {
    uint n_iters = p.padded_cols >> 8u;
    float nw[16];
    for (uint it = 0u; it < n_iters; ++it) {
        uint col = (it << 8u) + tid;
        nw[it] = (col < p.rows) ? norm_w[col] : 0.0f;
    }
    if (tid == 0u) {
        while (atomic_load_explicit(&sync_cnt[p.mat_sync_slot], memory_order_relaxed) < p.next_padded_cols) {}
        atomic_store_explicit(&sync_cnt[p.mat_sync_slot], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vxt_device_fence();
    float reg_v[16];
    float sq = 0.0f;
    for (uint it = 0u; it < n_iters; ++it) {
        uint col = (it << 8u) + tid;
        float v = (col < p.rows) ? cur_in[col] : 0.0f;
        reg_v[it] = v;
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    threadgroup float sg_tmp[8];
    if (lane == 0u) sg_tmp[sg_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float s8 = (lane < 8u) ? sg_tmp[lane] : 0.0f;
        s8 = simd_sum(s8);
        if (lane == 0u) sg_tmp[0] = rsqrt(s8 / float(p.rows) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = sg_tmp[0];
    bool write_f = (p.flags & 1u) != 0u;
    for (uint it = 0u; it < n_iters; ++it) {
        uint col = (it << 8u) + tid;
        float scaled = (col < p.rows) ? vxt_apply_col_sign(reg_v[it] * inv_rms * nw[it], col) : 0.0f;
        float rot = vxt_fwht32_lane(scaled, lane);
        x_rot_h_out[col] = half(rot * 0.0625f);
        if (write_f) x_rot_out[col] = rot;
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_m1_t2_1sg(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    VXT_COHERENT_DEV float* y       [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    device atomic_uint* sync_cnt    [[buffer(6)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half4 lut256[256];
    lut256[tid] = lut_bank[1024u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile = tg_pos.x * 8u + sg_id;
    if ((tile << 2u) < p.rows) {
        uint steps = p.padded_cols >> 8u;
        uint gh = lane >> 4u;
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint2* pw0 = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);
        float4 acc0 = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            acc0 = vxt_eval_t2_lut(
                lut256, pw0[(step << 5u) + lane],
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
        float4 sum0 = simd_sum(acc0) * 16.0f;
        if (lane == 0u) vxt_accum_tile4_coh(y, p.rows, tile, sum0, add_res);
    }
    vxt_signal_tg_done(sync_cnt, p.flags, p.sync_slot, tid);
}

[[max_total_threads_per_threadgroup(128)]]
kernel void vxt_m1_t2_sk2(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    VXT_COHERENT_DEV float* y       [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    device atomic_uint* sync_cnt    [[buffer(6)]],
    device const float* norm_w      [[buffer(7)]],
    device half* x_rot_h_out        [[buffer(8)]],
    device float* x_rot_out         [[buffer(9)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half4 lut256[256];
    threadgroup float4 sk_part[2];
    lut256[tid] = lut_bank[1024u + tid];
    lut256[128u + tid] = lut_bank[1152u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub = sg_id >> 1u;
    uint k_par = sg_id & 1u;
    uint tile = tg_pos.x * 2u + sub;
    uint steps = p.padded_cols >> 8u;
    uint h_s = (steps + 1u) >> 1u;
    uint s_beg = min(k_par * h_s, steps);
    uint s_end = min(s_beg + h_s, steps);
    uint gh = lane >> 4u;
    float4 acc0 = float4(0.0f);
    if ((tile << 2u) < p.rows) {
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint2* pw0 = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);
        for (uint step = s_beg; step < s_end; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            acc0 = vxt_eval_t2_lut(
                lut256, pw0[(step << 5u) + lane],
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
    }
    float4 sum0 = simd_sum(acc0) * 16.0f;
    if (k_par == 1u && lane == 0u) sk_part[sub] = sum0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
        vxt_accum_tile4_coh(y, p.rows, tile, sum0 + sk_part[sub], add_res);
    }
    vxt_epilogue_last_tg_norm(y, norm_w, x_rot_h_out, x_rot_out, sync_cnt, (threadgroup float*)lut256, p, 128u, tid, sg_id, lane);
}

kernel void vxt_m1_t2_sk4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    VXT_COHERENT_DEV float* y       [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    device atomic_uint* sync_cnt    [[buffer(6)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half4 lut256[256];
    threadgroup float4 sk_part[3];
    lut256[tid] = lut_bank[1024u + tid];
    lut256[128u + tid] = lut_bank[1152u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile = tg_pos.x;
    uint steps = p.padded_cols >> 8u;
    uint q_s = (steps + 3u) >> 2u;
    uint s_beg = min(sg_id * q_s, steps);
    uint s_end = min(s_beg + q_s, steps);
    uint gh = lane >> 4u;
    float4 acc0 = float4(0.0f);
    if ((tile << 2u) < p.rows) {
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint2* pw0 = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);
        for (uint step = s_beg; step < s_end; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            acc0 = vxt_eval_t2_lut(
                lut256, pw0[(step << 5u) + lane],
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
    }
    float4 sum0 = simd_sum(acc0) * 16.0f;
    if (sg_id > 0u && lane == 0u) sk_part[sg_id - 1u] = sum0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u && lane == 0u && (tile << 2u) < p.rows) {
        vxt_accum_tile4_coh(y, p.rows, tile, sum0 + sk_part[0] + sk_part[1] + sk_part[2], add_res);
    }
    vxt_signal_tg_done(sync_cnt, p.flags, p.sync_slot, tid);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_m1_t25_1sg(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    VXT_COHERENT_DEV float* y       [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    device atomic_uint* sync_cnt    [[buffer(6)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half4 lut1k[1024];
    for (uint i = tid; i < 1024u; i += 256u) lut1k[i] = lut_bank[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile = tg_pos.x * 8u + sg_id;
    if ((tile << 2u) < p.rows) {
        uint steps = p.padded_cols >> 8u;
        uint gh = lane >> 4u;
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 80ul;
        float4 acc0 = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            device const uint* p0 = pw0 + step * 80u;
            acc0 = vxt_eval_t25_lut1k(
                lut1k, ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
        float4 sum0 = simd_sum(acc0) * 16.0f;
        if (lane == 0u) vxt_accum_tile4_coh(y, p.rows, tile, sum0, add_res);
    }
    vxt_signal_tg_done(sync_cnt, p.flags, p.sync_slot, tid);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_m1_t25_sk2(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    VXT_COHERENT_DEV float* y       [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    device atomic_uint* sync_cnt    [[buffer(6)]],
    device const float* norm_w      [[buffer(7)]],
    device half* x_rot_h_out        [[buffer(8)]],
    device float* x_rot_out         [[buffer(9)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half4 lut1k[1024];
    for (uint i = tid; i < 1024u; i += 256u) lut1k[i] = lut_bank[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub = sg_id >> 1u;
    uint k_par = sg_id & 1u;
    uint tile = tg_pos.x * 4u + sub;
    uint steps = p.padded_cols >> 8u;
    uint h_s = (steps + 1u) >> 1u;
    uint s_beg = min(k_par * h_s, steps);
    uint s_end = min(s_beg + h_s, steps);
    uint gh = lane >> 4u;
    float4 acc0 = float4(0.0f);
    if ((tile << 2u) < p.rows) {
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 80ul;
        for (uint step = s_beg; step < s_end; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            device const uint* p0 = pw0 + step * 80u;
            acc0 = vxt_eval_t25_lut1k(
                lut1k, ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
    }
    float4 sum0 = simd_sum(acc0) * 16.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float4* sk_part = (threadgroup float4*)lut1k;
    if (k_par == 1u && lane == 0u) sk_part[sub] = sum0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
        vxt_accum_tile4_coh(y, p.rows, tile, sum0 + sk_part[sub], add_res);
    }
    vxt_epilogue_last_tg_norm(y, norm_w, x_rot_h_out, x_rot_out, sync_cnt, (threadgroup float*)lut1k + 32, p, 256u, tid, sg_id, lane);
}

[[max_total_threads_per_threadgroup(512)]]
kernel void vxt_m1_t25_sk4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    VXT_COHERENT_DEV float* y       [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    device atomic_uint* sync_cnt    [[buffer(6)]],
    device const float* norm_w      [[buffer(7)]],
    device half* x_rot_h_out        [[buffer(8)]],
    device float* x_rot_out         [[buffer(9)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half4 lut1k[1024];
    for (uint i = tid; i < 1024u; i += 512u) lut1k[i] = lut_bank[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub = sg_id >> 1u;
    uint k_par = sg_id & 1u;
    uint tile = tg_pos.x * 8u + sub;
    uint steps = p.padded_cols >> 8u;
    uint h_s = (steps + 1u) >> 1u;
    uint s_beg = min(k_par * h_s, steps);
    uint s_end = min(s_beg + h_s, steps);
    uint gh = lane >> 4u;
    float4 acc0 = float4(0.0f);
    if ((tile << 2u) < p.rows) {
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 80ul;
        for (uint step = s_beg; step < s_end; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            device const uint* p0 = pw0 + step * 80u;
            acc0 = vxt_eval_t25_lut1k(
                lut1k, ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
    }
    float4 sum0 = simd_sum(acc0) * 16.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float4* sk_part = (threadgroup float4*)lut1k;
    if (k_par == 1u && lane == 0u) sk_part[sub] = sum0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
        vxt_accum_tile4_coh(y, p.rows, tile, sum0 + sk_part[sub], add_res);
    }
    vxt_epilogue_last_tg_norm(y, norm_w, x_rot_h_out, x_rot_out, sync_cnt, (threadgroup float*)lut1k + 64, p, 512u, tid, sg_id, lane);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_m1_t3_1sg(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    VXT_COHERENT_DEV float* y       [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    device atomic_uint* sync_cnt    [[buffer(6)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool is_ec = (p.flags & 256u) != 0u;
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half2 lut64[64];
    if (tid < 64u) lut64[tid] = ((device const half2*)(lut_bank + (is_ec ? 1312u : 1280u)))[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile = tg_pos.x * 8u + sg_id;
    if ((tile << 2u) < p.rows) {
        uint steps = p.padded_cols >> 8u;
        uint gh = lane >> 4u;
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 96ul;
        float4 acc0 = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            device const uint* p0 = pw0 + step * 96u;
            acc0 = vxt_eval_t3_lut(
                lut64, ((device const uint2*)p0)[lane], p0[64u + lane],
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
        float4 sum0 = simd_sum(acc0) * 16.0f;
        if (lane == 0u) vxt_accum_tile4_coh(y, p.rows, tile, sum0, add_res);
    }
    vxt_signal_tg_done(sync_cnt, p.flags, p.sync_slot, tid);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_m1_t3_sk2(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    VXT_COHERENT_DEV float* y       [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    device atomic_uint* sync_cnt    [[buffer(6)]],
    device const float* norm_w      [[buffer(7)]],
    device half* x_rot_h_out        [[buffer(8)]],
    device float* x_rot_out         [[buffer(9)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool is_ec = (p.flags & 256u) != 0u;
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half2 lut64[64];
    threadgroup float4 sk_part[4];
    if (tid < 64u) lut64[tid] = ((device const half2*)(lut_bank + (is_ec ? 1312u : 1280u)))[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub = sg_id >> 1u;
    uint k_par = sg_id & 1u;
    uint tile = tg_pos.x * 4u + sub;
    uint steps = p.padded_cols >> 8u;
    uint h_s = (steps + 1u) >> 1u;
    uint s_beg = min(k_par * h_s, steps);
    uint s_end = min(s_beg + h_s, steps);
    uint gh = lane >> 4u;
    float4 acc0 = float4(0.0f);
    if ((tile << 2u) < p.rows) {
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 96ul;
        for (uint step = s_beg; step < s_end; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            device const uint* p0 = pw0 + step * 96u;
            acc0 = vxt_eval_t3_lut(
                lut64, ((device const uint2*)p0)[lane], p0[64u + lane],
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
    }
    float4 sum0 = simd_sum(acc0) * 16.0f;
    if (k_par == 1u && lane == 0u) sk_part[sub] = sum0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
        vxt_accum_tile4_coh(y, p.rows, tile, sum0 + sk_part[sub], add_res);
    }
    vxt_epilogue_last_tg_norm(y, norm_w, x_rot_h_out, x_rot_out, sync_cnt, (threadgroup float*)lut64, p, 256u, tid, sg_id, lane);
}

[[max_total_threads_per_threadgroup(512)]]
kernel void vxt_m1_t3_sk4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    VXT_COHERENT_DEV float* y       [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    device atomic_uint* sync_cnt    [[buffer(6)]],
    device const float* norm_w      [[buffer(7)]],
    device half* x_rot_h_out        [[buffer(8)]],
    device float* x_rot_out         [[buffer(9)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool is_ec = (p.flags & 256u) != 0u;
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half2 lut64[64];
    threadgroup float4 sk_part[8];
    if (tid < 64u) lut64[tid] = ((device const half2*)(lut_bank + (is_ec ? 1312u : 1280u)))[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sub = sg_id >> 1u;
    uint k_par = sg_id & 1u;
    uint tile = tg_pos.x * 8u + sub;
    uint steps = p.padded_cols >> 8u;
    uint h_s = (steps + 1u) >> 1u;
    uint s_beg = min(k_par * h_s, steps);
    uint s_end = min(s_beg + h_s, steps);
    uint gh = lane >> 4u;
    float4 acc0 = float4(0.0f);
    if ((tile << 2u) < p.rows) {
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 96ul;
        for (uint step = s_beg; step < s_end; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            device const uint* p0 = pw0 + step * 96u;
            acc0 = vxt_eval_t3_lut(
                lut64, ((device const uint2*)p0)[lane], p0[64u + lane],
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
    }
    float4 sum0 = simd_sum(acc0) * 16.0f;
    if (k_par == 1u && lane == 0u) sk_part[sub] = sum0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
        vxt_accum_tile4_coh(y, p.rows, tile, sum0 + sk_part[sub], add_res);
    }
    vxt_epilogue_last_tg_norm(y, norm_w, x_rot_h_out, x_rot_out, sync_cnt, (threadgroup float*)lut64, p, 512u, tid, sg_id, lane);
}

inline void vxt_m1_ffn_gu_finish_sk2(
    uint blk,
    uint n_ff,
    float4 g4,
    float4 u4,
    threadgroup float2* tg_gu,
    device half* x_rot_h_out,
    uint sg_id,
    uint lane
) {
    uint sub = sg_id >> 1u;
    uint k_par = sg_id & 1u;
    if (k_par == 1u && lane < 4u) tg_gu[sub * 4u + lane] = float2(g4[lane], u4[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_par == 0u && lane < 4u) tg_gu[sub * 4u + lane] += float2(g4[lane], u4[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        uint col = (blk << 5u) + lane;
        float2 gu = tg_gu[lane];
        float sw = (col < n_ff) ? vxt_apply_col_sign((gu.x / (1.0f + exp(-gu.x))) * gu.y, col) : 0.0f;
        float rot = vxt_fwht32_lane(sw, lane);
        x_rot_h_out[col] = half(rot * 0.0625f);
    }
}

[[max_total_threads_per_threadgroup(512)]]
kernel void vxt_m1_ffn_gu_t2(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    device half* x_rot_h_out        [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint blk = tg_pos.x;
    uint base_col = blk << 5u;
    if (base_col >= p.next_padded_cols) return;
    if (base_col >= p.cols) {
        if (tid < 32u) x_rot_h_out[base_col + tid] = half(0.0h);
        return;
    }
    threadgroup half4 lut256[256];
    threadgroup float2 tg_gu[32];
    if (tid < 256u) lut256[tid] = lut_bank[1024u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint h_s = (steps + 1u) >> 1u;
    uint sub = sg_id >> 1u;
    uint k_par = sg_id & 1u;
    uint s_beg = min(k_par * h_s, steps);
    uint s_end = min(s_beg + h_s, steps);
    uint gh = lane >> 4u;
    uint tile_g = blk * 16u + sub;
    uint tile_u = tile_g + 8u;
    device const ushort* sc_g = scales_f16 + ulong(tile_g) * ulong(steps) * 8ul;
    device const ushort* sc_u = scales_f16 + ulong(tile_u) * ulong(steps) * 8ul;
    device const uint2* pw_g = (device const uint2*)(packed_words + ulong(tile_g) * ulong(steps) * 64ul);
    device const uint2* pw_u = (device const uint2*)(packed_words + ulong(tile_u) * ulong(steps) * 64ul);
    float4 acc_g = float4(0.0f), acc_u = float4(0.0f);
    for (uint step = s_beg; step < s_end; ++step) {
        device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
        uint off = (step << 5u) + lane;
        uint soff = (step << 3u) + (gh << 2u);
        acc_g = vxt_eval_t2_lut(lut256, pw_g[off], *(device const half4*)(sc_g + soff), xp4[0], xp4[1], acc_g);
        acc_u = vxt_eval_t2_lut(lut256, pw_u[off], *(device const half4*)(sc_u + soff), xp4[0], xp4[1], acc_u);
    }
    vxt_m1_ffn_gu_finish_sk2(blk, p.cols, simd_sum(acc_g) * 16.0f, simd_sum(acc_u) * 16.0f, tg_gu, x_rot_h_out, sg_id, lane);
}

[[max_total_threads_per_threadgroup(512)]]
kernel void vxt_m1_ffn_gu_t25(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    device half* x_rot_h_out        [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint blk = tg_pos.x;
    uint base_col = blk << 5u;
    if (base_col >= p.next_padded_cols) return;
    if (base_col >= p.cols) {
        if (tid < 32u) x_rot_h_out[base_col + tid] = half(0.0h);
        return;
    }
    threadgroup half4 lut1k[1024];
    for (uint i = tid; i < 1024u; i += 512u) lut1k[i] = lut_bank[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint h_s = (steps + 1u) >> 1u;
    uint sub = sg_id >> 1u;
    uint k_par = sg_id & 1u;
    uint s_beg = min(k_par * h_s, steps);
    uint s_end = min(s_beg + h_s, steps);
    uint gh = lane >> 4u;
    uint tile_g = blk * 16u + sub;
    uint tile_u = tile_g + 8u;
    device const ushort* sc_g = scales_f16 + ulong(tile_g) * ulong(steps) * 8ul;
    device const ushort* sc_u = scales_f16 + ulong(tile_u) * ulong(steps) * 8ul;
    device const uint* pw_g = packed_words + ulong(tile_g) * ulong(steps) * 80ul;
    device const uint* pw_u = packed_words + ulong(tile_u) * ulong(steps) * 80ul;
    float4 acc_g = float4(0.0f), acc_u = float4(0.0f);
    for (uint step = s_beg; step < s_end; ++step) {
        device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
        device const uint* pg = pw_g + step * 80u;
        device const uint* pu = pw_u + step * 80u;
        uint soff = (step << 3u) + (gh << 2u);
        acc_g = vxt_eval_t25_lut1k(lut1k, ((device const uint2*)pg)[lane], uint(((device const ushort*)(pg + 64u))[lane]), *(device const half4*)(sc_g + soff), xp4[0], xp4[1], acc_g);
        acc_u = vxt_eval_t25_lut1k(lut1k, ((device const uint2*)pu)[lane], uint(((device const ushort*)(pu + 64u))[lane]), *(device const half4*)(sc_u + soff), xp4[0], xp4[1], acc_u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vxt_m1_ffn_gu_finish_sk2(blk, p.cols, simd_sum(acc_g) * 16.0f, simd_sum(acc_u) * 16.0f, (threadgroup float2*)lut1k, x_rot_h_out, sg_id, lane);
}

[[max_total_threads_per_threadgroup(512)]]
kernel void vxt_m1_ffn_gu_t3(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const half4* x_rot_h     [[buffer(2)]],
    device half* x_rot_h_out        [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const half4* lut_bank    [[buffer(5)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint blk = tg_pos.x;
    uint base_col = blk << 5u;
    if (base_col >= p.next_padded_cols) return;
    if (base_col >= p.cols) {
        if (tid < 32u) x_rot_h_out[base_col + tid] = half(0.0h);
        return;
    }
    bool is_ec = (p.flags & 256u) != 0u;
    threadgroup half2 lut64[64];
    threadgroup float2 tg_gu[32];
    if (tid < 64u) lut64[tid] = ((device const half2*)(lut_bank + (is_ec ? 1312u : 1280u)))[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint h_s = (steps + 1u) >> 1u;
    uint sub = sg_id >> 1u;
    uint k_par = sg_id & 1u;
    uint s_beg = min(k_par * h_s, steps);
    uint s_end = min(s_beg + h_s, steps);
    uint gh = lane >> 4u;
    uint tile_g = blk * 16u + sub;
    uint tile_u = tile_g + 8u;
    device const ushort* sc_g = scales_f16 + ulong(tile_g) * ulong(steps) * 8ul;
    device const ushort* sc_u = scales_f16 + ulong(tile_u) * ulong(steps) * 8ul;
    device const uint* pw_g = packed_words + ulong(tile_g) * ulong(steps) * 96ul;
    device const uint* pw_u = packed_words + ulong(tile_u) * ulong(steps) * 96ul;
    float4 acc_g = float4(0.0f), acc_u = float4(0.0f);
    for (uint step = s_beg; step < s_end; ++step) {
        device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
        device const uint* pg = pw_g + step * 96u;
        device const uint* pu = pw_u + step * 96u;
        uint soff = (step << 3u) + (gh << 2u);
        acc_g = vxt_eval_t3_lut(lut64, ((device const uint2*)pg)[lane], pg[64u + lane], *(device const half4*)(sc_g + soff), xp4[0], xp4[1], acc_g);
        acc_u = vxt_eval_t3_lut(lut64, ((device const uint2*)pu)[lane], pu[64u + lane], *(device const half4*)(sc_u + soff), xp4[0], xp4[1], acc_u);
    }
    vxt_m1_ffn_gu_finish_sk2(blk, p.cols, simd_sum(acc_g) * 16.0f, simd_sum(acc_u) * 16.0f, tg_gu, x_rot_h_out, sg_id, lane);
}

// 6. Dual-Pipe FP16 `M = 1` GEMV Kernels (`NR = 4` + 2/4-Way Split-K + Optional Zero-Atomic Residual & Norm Prologue)
inline void vxt_prepare_x_sh_128(
    bool fuse_norm,
    uint padded_cols,
    uint norm_dim,
    float norm_eps,
    device const float* cur_in,
    device const float* norm_w,
    device float* x_rot_pad,
    threadgroup half* x_sh,
    threadgroup float* sg_tmp,
    uint tid,
    uint sg_id,
    uint lane,
    bool write_global_rot
) {
    if (fuse_norm) {
        float sq = 0.0f;
        for (uint col = tid; col < norm_dim; col += 128u) {
            float v = cur_in[col];
            sq = fma(v, v, sq);
        }
        sq = simd_sum(sq);
        if (lane == 0u) sg_tmp[sg_id] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u) {
            float s = (lane < 4u) ? sg_tmp[lane] : 0.0f;
            s = simd_sum(s);
            if (lane == 0u) sg_tmp[0] = rsqrt(s / float(norm_dim) + norm_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = sg_tmp[0];
        uint n_blks = padded_cols >> 5u;
        for (uint blk = sg_id; blk < n_blks; blk += 4u) {
            uint col = (blk << 5u) + lane;
            float val = (col < norm_dim) ? vxt_apply_col_sign(cur_in[col] * inv_rms * norm_w[col], col) : 0.0f;
            float rot = vxt_fwht32_lane(val, lane);
            x_sh[col] = half(rot * 0.0625f);
            if (write_global_rot) x_rot_pad[col] = rot;
        }
    } else if (padded_cols <= 2048u) {
        for (uint col = tid; col < padded_cols; col += 128u) {
            x_sh[col] = half(x_rot_pad[col] * 0.0625f);
        }
    }
}

inline void vxt_prepare_x_sh_256(
    bool fuse_norm,
    uint padded_cols,
    uint norm_dim,
    float norm_eps,
    device const float* cur_in,
    device const float* norm_w,
    device const float* x_rot_in,
    threadgroup half* x_sh,
    threadgroup float* sg_tmp,
    uint tid,
    uint sg_id,
    uint lane
) {
    if (fuse_norm) {
        float sq = 0.0f;
        for (uint col = tid; col < norm_dim; col += 256u) {
            float v = cur_in[col];
            sq = fma(v, v, sq);
        }
        sq = simd_sum(sq);
        if (lane == 0u) sg_tmp[sg_id] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u) {
            float s = (lane < 8u) ? sg_tmp[lane] : 0.0f;
            s = simd_sum(s);
            if (lane == 0u) sg_tmp[0] = rsqrt(s / float(norm_dim) + norm_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = sg_tmp[0];
        uint n_blks = padded_cols >> 5u;
        for (uint blk = sg_id; blk < n_blks; blk += 8u) {
            uint col = (blk << 5u) + lane;
            float val = (col < norm_dim) ? vxt_apply_col_sign(cur_in[col] * inv_rms * norm_w[col], col) : 0.0f;
            x_sh[col] = half(vxt_fwht32_lane(val, lane) * 0.0625f);
        }
    } else if (padded_cols <= 2048u) {
        for (uint col = tid; col < padded_cols; col += 256u) {
            x_sh[col] = half(x_rot_in[col] * 0.0625f);
        }
    }
}

inline void vxt_load_xh_pair(
    bool use_sh,
    threadgroup const half* x_sh,
    device const float* x_rot_pad,
    uint step,
    uint lane,
    thread half4& xh0,
    thread half4& xh1
) {
    uint off = (step << 8u) + (lane << 3u);
    if (use_sh) {
        threadgroup const half4* xp = (threadgroup const half4*)(x_sh + off);
        xh0 = xp[0];
        xh1 = xp[1];
    } else {
        device const float4* xp = (device const float4*)(x_rot_pad + off);
        xh0 = half4(xp[0] * 0.0625f);
        xh1 = half4(xp[1] * 0.0625f);
    }
}

kernel void vxt_llm_turbo2_nr4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device float* x_rot_pad         [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const float* cur_in      [[buffer(5)]],
    device const float* norm_w      [[buffer(6)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint mode_tps = p.flags & 255u;
    bool add_res = (p.flags & 512u) != 0u;
    bool fuse_norm = (p.flags & 1024u) != 0u;
    bool use_sh = fuse_norm || (p.padded_cols <= 2048u);

    threadgroup half4 lut256[256];
    threadgroup half x_sh[2048];
    threadgroup float sg_tmp[4];
    vxt_init_lut256_n(lut256, tid, 128u);
    vxt_prepare_x_sh_128(
        fuse_norm, p.padded_cols, p.norm_dim, p.norm_eps,
        cur_in, norm_w, x_rot_pad, x_sh, sg_tmp, tid, sg_id, lane, tg_pos.x == 0u
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;

    if (mode_tps == 2u) {
        uint tile = tg_pos.x;
        uint q_s = (steps + 3u) >> 2u;
        uint s_beg = min(sg_id * q_s, steps);
        uint s_end = min(s_beg + q_s, steps);
        float4 acc0 = float4(0.0f);
        if ((tile << 2u) < p.rows) {
            device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
            device const uint2* pw0 = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);
            for (uint step = s_beg; step < s_end; ++step) {
                half4 xh0, xh1;
                vxt_load_xh_pair(use_sh, x_sh, x_rot_pad, step, lane, xh0, xh1);
                acc0 = vxt_eval_t2_lut(
                    lut256, pw0[(step << 5u) + lane],
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
                );
            }
        }
        float4 sum0 = simd_sum(acc0) * 16.0f;
        threadgroup float4 sk4_part[3];
        if (sg_id > 0u && lane == 0u) sk4_part[sg_id - 1u] = sum0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u && lane == 0u && (tile << 2u) < p.rows) {
            vxt_accum_tile4(y, p.rows, tile, sum0 + sk4_part[0] + sk4_part[1] + sk4_part[2], add_res);
        }
        return;
    }

    if (mode_tps == 0u) {
        uint tile = tg_pos.x * 2u + (sg_id >> 1u);
        uint k_par = sg_id & 1u;
        uint half_s = steps >> 1u;
        uint s_beg = (k_par == 0u) ? 0u : half_s;
        uint s_end = (k_par == 0u) ? half_s : steps;
        float4 acc0 = float4(0.0f);
        if ((tile << 2u) < p.rows) {
            device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
            device const uint2* pw0 = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);
            for (uint step = s_beg; step < s_end; ++step) {
                half4 xh0, xh1;
                vxt_load_xh_pair(use_sh, x_sh, x_rot_pad, step, lane, xh0, xh1);
                acc0 = vxt_eval_t2_lut(
                    lut256, pw0[(step << 5u) + lane],
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
                );
            }
        }
        float4 sum0 = simd_sum(acc0) * 16.0f;
        threadgroup float4 sk_part[2];
        if (k_par == 1u && lane == 0u) sk_part[sg_id >> 1u] = sum0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
            vxt_accum_tile4(y, p.rows, tile, sum0 + sk_part[sg_id >> 1u], add_res);
        }
        return;
    }

    uint tile = tg_pos.x * 4u + sg_id;
    if ((tile << 2u) >= p.rows) return;
    device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
    device const uint2* pw0 = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);
    float4 acc0 = float4(0.0f);
    for (uint step = 0u; step < steps; ++step) {
        half4 xh0, xh1;
        vxt_load_xh_pair(use_sh, x_sh, x_rot_pad, step, lane, xh0, xh1);
        acc0 = vxt_eval_t2_lut(
            lut256, pw0[(step << 5u) + lane],
            *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
        );
    }
    float4 sum0 = simd_sum(acc0) * 16.0f;
    if (lane == 0u) vxt_accum_tile4(y, p.rows, tile, sum0, add_res);
}

kernel void vxt_llm_turbo25_nr4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device float* x_rot_pad         [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const float* cur_in      [[buffer(5)]],
    device const float* norm_w      [[buffer(6)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint mode_tps = p.flags & 255u;
    bool add_res = (p.flags & 512u) != 0u;
    bool fuse_norm = (p.flags & 1024u) != 0u;
    bool use_sh = fuse_norm || (p.padded_cols <= 2048u);

    threadgroup half2 lut_lo[256];
    threadgroup half2 lut_hi[256];
    threadgroup half x_sh[2048];
    threadgroup float sg_tmp[4];
    vxt_init_lut256_t25(lut_lo, lut_hi, tid, 128u);
    vxt_prepare_x_sh_128(
        fuse_norm, p.padded_cols, p.norm_dim, p.norm_eps,
        cur_in, norm_w, x_rot_pad, x_sh, sg_tmp, tid, sg_id, lane, tg_pos.x == 0u
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;

    if (mode_tps == 2u) {
        uint tile = tg_pos.x;
        uint q_s = (steps + 3u) >> 2u;
        uint s_beg = min(sg_id * q_s, steps);
        uint s_end = min(s_beg + q_s, steps);
        float4 acc0 = float4(0.0f);
        if ((tile << 2u) < p.rows) {
            device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
            device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 80ul;
            for (uint step = s_beg; step < s_end; ++step) {
                half4 xh0, xh1;
                vxt_load_xh_pair(use_sh, x_sh, x_rot_pad, step, lane, xh0, xh1);
                device const uint* p0 = pw0 + step * 80u;
                acc0 = vxt_eval_t25_lut(
                    lut_lo, lut_hi, ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
                );
            }
        }
        float4 sum0 = simd_sum(acc0) * 16.0f;
        threadgroup float4 sk4_part[3];
        if (sg_id > 0u && lane == 0u) sk4_part[sg_id - 1u] = sum0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u && lane == 0u && (tile << 2u) < p.rows) {
            vxt_accum_tile4(y, p.rows, tile, sum0 + sk4_part[0] + sk4_part[1] + sk4_part[2], add_res);
        }
        return;
    }

    if (mode_tps == 0u) {
        uint tile = tg_pos.x * 2u + (sg_id >> 1u);
        uint k_par = sg_id & 1u;
        uint half_s = steps >> 1u;
        uint s_beg = (k_par == 0u) ? 0u : half_s;
        uint s_end = (k_par == 0u) ? half_s : steps;
        float4 acc0 = float4(0.0f);
        if ((tile << 2u) < p.rows) {
            device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
            device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 80ul;
            for (uint step = s_beg; step < s_end; ++step) {
                half4 xh0, xh1;
                vxt_load_xh_pair(use_sh, x_sh, x_rot_pad, step, lane, xh0, xh1);
                device const uint* p0 = pw0 + step * 80u;
                acc0 = vxt_eval_t25_lut(
                    lut_lo, lut_hi, ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
                );
            }
        }
        float4 sum0 = simd_sum(acc0) * 16.0f;
        threadgroup float4 sk_part[2];
        if (k_par == 1u && lane == 0u) sk_part[sg_id >> 1u] = sum0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
            vxt_accum_tile4(y, p.rows, tile, sum0 + sk_part[sg_id >> 1u], add_res);
        }
        return;
    }

    uint tile = tg_pos.x * 4u + sg_id;
    if ((tile << 2u) >= p.rows) return;
    device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
    device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 80ul;
    float4 acc0 = float4(0.0f);
    for (uint step = 0u; step < steps; ++step) {
        half4 xh0, xh1;
        vxt_load_xh_pair(use_sh, x_sh, x_rot_pad, step, lane, xh0, xh1);
        device const uint* p0 = pw0 + step * 80u;
        acc0 = vxt_eval_t25_lut(
            lut_lo, lut_hi, ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
            *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
        );
    }
    float4 sum0 = simd_sum(acc0) * 16.0f;
    if (lane == 0u) vxt_accum_tile4(y, p.rows, tile, sum0, add_res);
}

kernel void vxt_llm_turbo3_nr4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device float* x_rot_pad         [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    device const float* cur_in      [[buffer(5)]],
    device const float* norm_w      [[buffer(6)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool is_ec = (p.flags & 256u) != 0u;
    uint mode_tps = p.flags & 255u;
    bool add_res = (p.flags & 512u) != 0u;
    bool fuse_norm = (p.flags & 1024u) != 0u;
    bool use_sh = fuse_norm || (p.padded_cols <= 2048u);

    threadgroup half2 lut64[64];
    threadgroup half x_sh[2048];
    threadgroup float sg_tmp[4];
    vxt_init_lut64_sel(lut64, tid, is_ec);
    vxt_prepare_x_sh_128(
        fuse_norm, p.padded_cols, p.norm_dim, p.norm_eps,
        cur_in, norm_w, x_rot_pad, x_sh, sg_tmp, tid, sg_id, lane, tg_pos.x == 0u
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;

    if (mode_tps == 2u) {
        uint tile = tg_pos.x;
        uint q_s = (steps + 3u) >> 2u;
        uint s_beg = min(sg_id * q_s, steps);
        uint s_end = min(s_beg + q_s, steps);
        float4 acc0 = float4(0.0f);
        if ((tile << 2u) < p.rows) {
            device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
            device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 96ul;
            for (uint step = s_beg; step < s_end; ++step) {
                half4 xh0, xh1;
                vxt_load_xh_pair(use_sh, x_sh, x_rot_pad, step, lane, xh0, xh1);
                device const uint* p0 = pw0 + step * 96u;
                acc0 = vxt_eval_t3_lut(
                    lut64, ((device const uint2*)p0)[lane], p0[64u + lane],
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
                );
            }
        }
        float4 sum0 = simd_sum(acc0) * 16.0f;
        threadgroup float4 sk4_part[3];
        if (sg_id > 0u && lane == 0u) sk4_part[sg_id - 1u] = sum0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u && lane == 0u && (tile << 2u) < p.rows) {
            vxt_accum_tile4(y, p.rows, tile, sum0 + sk4_part[0] + sk4_part[1] + sk4_part[2], add_res);
        }
        return;
    }

    if (mode_tps == 0u) {
        uint tile = tg_pos.x * 2u + (sg_id >> 1u);
        uint k_par = sg_id & 1u;
        uint half_s = steps >> 1u;
        uint s_beg = (k_par == 0u) ? 0u : half_s;
        uint s_end = (k_par == 0u) ? half_s : steps;
        float4 acc0 = float4(0.0f);
        if ((tile << 2u) < p.rows) {
            device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
            device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 96ul;
            for (uint step = s_beg; step < s_end; ++step) {
                half4 xh0, xh1;
                vxt_load_xh_pair(use_sh, x_sh, x_rot_pad, step, lane, xh0, xh1);
                device const uint* p0 = pw0 + step * 96u;
                acc0 = vxt_eval_t3_lut(
                    lut64, ((device const uint2*)p0)[lane], p0[64u + lane],
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
                );
            }
        }
        float4 sum0 = simd_sum(acc0) * 16.0f;
        threadgroup float4 sk_part[2];
        if (k_par == 1u && lane == 0u) sk_part[sg_id >> 1u] = sum0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
            vxt_accum_tile4(y, p.rows, tile, sum0 + sk_part[sg_id >> 1u], add_res);
        }
        return;
    }

    uint tile = tg_pos.x * 4u + sg_id;
    if ((tile << 2u) >= p.rows) return;
    device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
    device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 96ul;
    float4 acc0 = float4(0.0f);
    for (uint step = 0u; step < steps; ++step) {
        half4 xh0, xh1;
        vxt_load_xh_pair(use_sh, x_sh, x_rot_pad, step, lane, xh0, xh1);
        device const uint* p0 = pw0 + step * 96u;
        acc0 = vxt_eval_t3_lut(
            lut64, ((device const uint2*)p0)[lane], p0[64u + lane],
            *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
        );
    }
    float4 sum0 = simd_sum(acc0) * 16.0f;
    if (lane == 0u) vxt_accum_tile4(y, p.rows, tile, sum0, add_res);
}

kernel void vxt_llm_turbo2_nr8(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const float* x_rot_pad   [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint tps = p.flags & 255u;
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half4 lut256[256];
    vxt_init_lut256_n(lut256, tid, 128u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;
    uint sg_pair_base = (tg_pos.x * 4u + sg_id) * tps;

    for (uint kp = 0u; kp < tps; ++kp) {
        uint tile0 = (sg_pair_base + kp) * 2u;
        if ((tile0 << 2u) >= p.rows) break;
        bool has_t1 = ((tile0 + 1u) << 2u) < p.rows;

        device const ushort* sc0 = scales_f16 + ulong(tile0) * ulong(steps) * 8ul;
        device const uint2* pw0 = (device const uint2*)(packed_words + ulong(tile0) * ulong(steps) * 64ul);
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);

        if (has_t1) {
            device const ushort* sc1 = sc0 + ulong(steps) * 8ul;
            device const uint2* pw1 = pw0 + ulong(steps) * 32ul;
            for (uint step = 0u; step < steps; ++step) {
                device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
                half4 xh0 = half4(xp4[0] * 0.0625f), xh1 = half4(xp4[1] * 0.0625f);
                uint off = (step << 5u) + lane;
                uint soff = (step << 3u) + (gh << 2u);
                acc0 = vxt_eval_t2_lut(lut256, pw0[off], *(device const half4*)(sc0 + soff), xh0, xh1, acc0);
                acc1 = vxt_eval_t2_lut(lut256, pw1[off], *(device const half4*)(sc1 + soff), xh0, xh1, acc1);
            }
        } else {
            for (uint step = 0u; step < steps; ++step) {
                device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
                half4 xh0 = half4(xp4[0] * 0.0625f), xh1 = half4(xp4[1] * 0.0625f);
                acc0 = vxt_eval_t2_lut(
                    lut256, pw0[(step << 5u) + lane],
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
                );
            }
        }

        float4 sum0 = simd_sum(acc0) * 16.0f;
        float4 sum1 = simd_sum(acc1) * 16.0f;
        if (lane == 0u) {
            vxt_accum_tile4(y, p.rows, tile0, sum0, add_res);
            if (has_t1) vxt_accum_tile4(y, p.rows, tile0 + 1u, sum1, add_res);
        }
    }
}

kernel void vxt_llm_turbo3_nr8(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const float* x_rot_pad   [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool is_ec = (p.flags & 256u) != 0u;
    uint tps = p.flags & 255u;
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half2 lut64[64];
    vxt_init_lut64_sel(lut64, tid, is_ec);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;
    uint sg_pair_base = (tg_pos.x * 4u + sg_id) * tps;

    for (uint kp = 0u; kp < tps; ++kp) {
        uint tile0 = (sg_pair_base + kp) * 2u;
        if ((tile0 << 2u) >= p.rows) break;
        bool has_t1 = ((tile0 + 1u) << 2u) < p.rows;

        device const ushort* sc0 = scales_f16 + ulong(tile0) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile0) * ulong(steps) * 96ul;
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);

        if (has_t1) {
            device const ushort* sc1 = sc0 + ulong(steps) * 8ul;
            device const uint* pw1 = pw0 + ulong(steps) * 96ul;
            for (uint step = 0u; step < steps; ++step) {
                device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
                half4 xh0 = half4(xp4[0] * 0.0625f), xh1 = half4(xp4[1] * 0.0625f);
                device const uint* p0 = pw0 + step * 96u;
                device const uint* p1 = pw1 + step * 96u;
                uint soff = (step << 3u) + (gh << 2u);
                acc0 = vxt_eval_t3_lut(
                    lut64, ((device const uint2*)p0)[lane], p0[64u + lane],
                    *(device const half4*)(sc0 + soff), xh0, xh1, acc0
                );
                acc1 = vxt_eval_t3_lut(
                    lut64, ((device const uint2*)p1)[lane], p1[64u + lane],
                    *(device const half4*)(sc1 + soff), xh0, xh1, acc1
                );
            }
        } else {
            for (uint step = 0u; step < steps; ++step) {
                device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
                half4 xh0 = half4(xp4[0] * 0.0625f), xh1 = half4(xp4[1] * 0.0625f);
                device const uint* p0 = pw0 + step * 96u;
                acc0 = vxt_eval_t3_lut(
                    lut64, ((device const uint2*)p0)[lane], p0[64u + lane],
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
                );
            }
        }

        float4 sum0 = simd_sum(acc0) * 16.0f;
        float4 sum1 = simd_sum(acc1) * 16.0f;
        if (lane == 0u) {
            vxt_accum_tile4(y, p.rows, tile0, sum0, add_res);
            if (has_t1) vxt_accum_tile4(y, p.rows, tile0 + 1u, sum1, add_res);
        }
    }
}

kernel void vxt_llm_turbo25_nr8(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const float* x_rot_pad   [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboLlmMatParams& p   [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint tps = p.flags & 255u;
    bool add_res = (p.flags & 512u) != 0u;
    threadgroup half2 lut_lo[256];
    threadgroup half2 lut_hi[256];
    vxt_init_lut256_t25(lut_lo, lut_hi, tid, 128u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;
    uint sg_pair_base = (tg_pos.x * 4u + sg_id) * tps;

    for (uint kp = 0u; kp < tps; ++kp) {
        uint tile0 = (sg_pair_base + kp) * 2u;
        if ((tile0 << 2u) >= p.rows) break;
        bool has_t1 = ((tile0 + 1u) << 2u) < p.rows;

        device const ushort* sc0 = scales_f16 + ulong(tile0) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile0) * ulong(steps) * 80ul;
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);

        if (has_t1) {
            device const ushort* sc1 = sc0 + ulong(steps) * 8ul;
            device const uint* pw1 = pw0 + ulong(steps) * 80ul;
            for (uint step = 0u; step < steps; ++step) {
                device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
                half4 xh0 = half4(xp4[0] * 0.0625f), xh1 = half4(xp4[1] * 0.0625f);
                device const uint* p0 = pw0 + step * 80u;
                device const uint* p1 = pw1 + step * 80u;
                uint soff = (step << 3u) + (gh << 2u);
                acc0 = vxt_eval_t25_lut(
                    lut_lo, lut_hi, ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                    *(device const half4*)(sc0 + soff), xh0, xh1, acc0
                );
                acc1 = vxt_eval_t25_lut(
                    lut_lo, lut_hi, ((device const uint2*)p1)[lane], uint(((device const ushort*)(p1 + 64u))[lane]),
                    *(device const half4*)(sc1 + soff), xh0, xh1, acc1
                );
            }
        } else {
            for (uint step = 0u; step < steps; ++step) {
                device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
                half4 xh0 = half4(xp4[0] * 0.0625f), xh1 = half4(xp4[1] * 0.0625f);
                device const uint* p0 = pw0 + step * 80u;
                acc0 = vxt_eval_t25_lut(
                    lut_lo, lut_hi, ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xh0, xh1, acc0
                );
            }
        }

        float4 sum0 = simd_sum(acc0) * 16.0f;
        float4 sum1 = simd_sum(acc1) * 16.0f;
        if (lane == 0u) {
            vxt_accum_tile4(y, p.rows, tile0, sum0, add_res);
            if (has_t1) vxt_accum_tile4(y, p.rows, tile0 + 1u, sum1, add_res);
        }
    }
}

// 6b. Fused `post_attn_norm` + `w_ffn_gu` + `SwiGLU` + `FWHT32` (`M = 1`)
inline void vxt_ffn_gu_epilogue(
    uint blk,
    uint ks,
    float4 g4,
    float4 u4,
    threadgroup float2* tg_gu,
    device float* x_rot_out,
    VXT_COHERENT_DEV float2* gu_part,
    device atomic_uint* sync_cnt,
    constant TurboFfnFusedParams& p,
    uint sg_id,
    uint lane
) {
    uint base_col = blk << 5u;
    if (lane < 4u) {
        tg_gu[sg_id * 4u + lane] = float2(g4[lane], u4[lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id != 0u) return;

    if (p.k_splits <= 1u) {
        float2 s = tg_gu[lane];
        uint col_l = base_col + lane;
        float sw = (col_l < p.n_ff)
            ? vxt_apply_col_sign((s.x / (1.0f + exp(-s.x))) * s.y, col_l)
            : 0.0f;
        x_rot_out[col_l] = vxt_fwht32_lane(sw, lane);
        return;
    }

    gu_part[(blk * p.k_splits + ks) * 32u + lane] = tg_gu[lane];
    vxt_device_fence();
    uint prev = 0u;
    if (lane == 0u) {
        prev = atomic_fetch_add_explicit(&sync_cnt[p.sync_base + blk], 1u, memory_order_relaxed);
    }
    prev = simd_broadcast_first(prev);
    if (prev + 1u == p.k_splits) {
        vxt_device_fence();
        if (lane == 0u) {
            atomic_store_explicit(&sync_cnt[p.sync_base + blk], 0u, memory_order_relaxed);
        }
        uint base_p = blk * p.k_splits * 32u + lane;
        float2 s = float2(0.0f);
        for (uint c = 0u; c < p.k_splits; ++c) {
            s += gu_part[base_p + c * 32u];
        }
        uint col_l = base_col + lane;
        float sw = (col_l < p.n_ff)
            ? vxt_apply_col_sign((s.x / (1.0f + exp(-s.x))) * s.y, col_l)
            : 0.0f;
        x_rot_out[col_l] = vxt_fwht32_lane(sw, lane);
    }
}

kernel void vxt_llm_ffn_gu_swiglu_t2(
    device const ushort* scales_f16     [[buffer(0)]],
    device const uint* packed_words     [[buffer(1)]],
    device const float* x_rot_in        [[buffer(2)]],
    device float* x_rot_out             [[buffer(3)]],
    constant TurboFfnFusedParams& p     [[buffer(4)]],
    VXT_COHERENT_DEV float2* gu_part    [[buffer(5)]],
    device atomic_uint* sync_cnt        [[buffer(6)]],
    device const float* cur_in          [[buffer(7)]],
    device const float* norm_w          [[buffer(8)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint blk = tg_pos.x;
    uint ks = tg_pos.y;
    uint base_col = blk << 5u;
    if (base_col >= p.padded_ff || ks >= max(p.k_splits, 1u)) return;
    if (base_col >= p.n_ff) {
        if (ks == 0u && tid < 32u) x_rot_out[base_col + tid] = 0.0f;
        return;
    }

    bool fuse_norm = (p.fuse_norm != 0u);
    bool use_sh = fuse_norm || (p.padded_cols <= 2048u);
    threadgroup half4 lut256[256];
    threadgroup half x_sh[2048];
    threadgroup float2 tg_gu[32];
    vxt_init_lut256_n(lut256, tid, 256u);
    vxt_prepare_x_sh_256(
        fuse_norm, p.padded_cols, p.norm_dim, p.norm_eps,
        cur_in, norm_w, x_rot_in, x_sh, (threadgroup float*)tg_gu, tid, sg_id, lane
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint k_sp = max(p.k_splits, 1u);
    uint q_s = (steps + k_sp - 1u) / k_sp;
    uint s_beg = min(ks * q_s, steps);
    uint s_end = min(s_beg + q_s, steps);
    uint gh = lane >> 4u;
    uint tile_g = blk * 16u + sg_id;
    uint tile_u = tile_g + 8u;
    device const ushort* sc_g = scales_f16 + ulong(tile_g) * ulong(steps) * 8ul;
    device const ushort* sc_u = scales_f16 + ulong(tile_u) * ulong(steps) * 8ul;
    device const uint2* pw_g = (device const uint2*)(packed_words + ulong(tile_g) * ulong(steps) * 64ul);
    device const uint2* pw_u = (device const uint2*)(packed_words + ulong(tile_u) * ulong(steps) * 64ul);

    float4 acc_g = float4(0.0f);
    float4 acc_u = float4(0.0f);
    for (uint step = s_beg; step < s_end; ++step) {
        half4 xh0, xh1;
        vxt_load_xh_pair(use_sh, x_sh, x_rot_in, step, lane, xh0, xh1);
        uint off = (step << 5u) + lane;
        uint soff = (step << 3u) + (gh << 2u);
        acc_g = vxt_eval_t2_lut(lut256, pw_g[off], *(device const half4*)(sc_g + soff), xh0, xh1, acc_g);
        acc_u = vxt_eval_t2_lut(lut256, pw_u[off], *(device const half4*)(sc_u + soff), xh0, xh1, acc_u);
    }

    vxt_ffn_gu_epilogue(
        blk, ks, simd_sum(acc_g) * 16.0f, simd_sum(acc_u) * 16.0f,
        tg_gu, x_rot_out, gu_part, sync_cnt, p, sg_id, lane
    );
}

kernel void vxt_llm_ffn_gu_swiglu_t25(
    device const ushort* scales_f16     [[buffer(0)]],
    device const uint* packed_words     [[buffer(1)]],
    device const float* x_rot_in        [[buffer(2)]],
    device float* x_rot_out             [[buffer(3)]],
    constant TurboFfnFusedParams& p     [[buffer(4)]],
    VXT_COHERENT_DEV float2* gu_part    [[buffer(5)]],
    device atomic_uint* sync_cnt        [[buffer(6)]],
    device const float* cur_in          [[buffer(7)]],
    device const float* norm_w          [[buffer(8)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint blk = tg_pos.x;
    uint ks = tg_pos.y;
    uint base_col = blk << 5u;
    if (base_col >= p.padded_ff || ks >= max(p.k_splits, 1u)) return;
    if (base_col >= p.n_ff) {
        if (ks == 0u && tid < 32u) x_rot_out[base_col + tid] = 0.0f;
        return;
    }

    bool fuse_norm = (p.fuse_norm != 0u);
    bool use_sh = fuse_norm || (p.padded_cols <= 2048u);
    threadgroup half2 lut_lo[256];
    threadgroup half2 lut_hi[256];
    threadgroup half x_sh[2048];
    threadgroup float2 tg_gu[32];
    vxt_init_lut256_t25(lut_lo, lut_hi, tid, 256u);
    vxt_prepare_x_sh_256(
        fuse_norm, p.padded_cols, p.norm_dim, p.norm_eps,
        cur_in, norm_w, x_rot_in, x_sh, (threadgroup float*)tg_gu, tid, sg_id, lane
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint k_sp = max(p.k_splits, 1u);
    uint q_s = (steps + k_sp - 1u) / k_sp;
    uint s_beg = min(ks * q_s, steps);
    uint s_end = min(s_beg + q_s, steps);
    uint gh = lane >> 4u;
    uint tile_g = blk * 16u + sg_id;
    uint tile_u = tile_g + 8u;
    device const ushort* sc_g = scales_f16 + ulong(tile_g) * ulong(steps) * 8ul;
    device const ushort* sc_u = scales_f16 + ulong(tile_u) * ulong(steps) * 8ul;
    device const uint* pw_g = packed_words + ulong(tile_g) * ulong(steps) * 80ul;
    device const uint* pw_u = packed_words + ulong(tile_u) * ulong(steps) * 80ul;

    float4 acc_g = float4(0.0f);
    float4 acc_u = float4(0.0f);
    for (uint step = s_beg; step < s_end; ++step) {
        half4 xh0, xh1;
        vxt_load_xh_pair(use_sh, x_sh, x_rot_in, step, lane, xh0, xh1);
        device const uint* pg = pw_g + step * 80u;
        device const uint* pu = pw_u + step * 80u;
        uint soff = (step << 3u) + (gh << 2u);
        acc_g = vxt_eval_t25_lut(
            lut_lo, lut_hi, ((device const uint2*)pg)[lane], uint(((device const ushort*)(pg + 64u))[lane]),
            *(device const half4*)(sc_g + soff), xh0, xh1, acc_g
        );
        acc_u = vxt_eval_t25_lut(
            lut_lo, lut_hi, ((device const uint2*)pu)[lane], uint(((device const ushort*)(pu + 64u))[lane]),
            *(device const half4*)(sc_u + soff), xh0, xh1, acc_u
        );
    }

    vxt_ffn_gu_epilogue(
        blk, ks, simd_sum(acc_g) * 16.0f, simd_sum(acc_u) * 16.0f,
        tg_gu, x_rot_out, gu_part, sync_cnt, p, sg_id, lane
    );
}

kernel void vxt_llm_ffn_gu_swiglu_t3(
    device const ushort* scales_f16     [[buffer(0)]],
    device const uint* packed_words     [[buffer(1)]],
    device const float* x_rot_in        [[buffer(2)]],
    device float* x_rot_out             [[buffer(3)]],
    constant TurboFfnFusedParams& p     [[buffer(4)]],
    VXT_COHERENT_DEV float2* gu_part    [[buffer(5)]],
    device atomic_uint* sync_cnt        [[buffer(6)]],
    device const float* cur_in          [[buffer(7)]],
    device const float* norm_w          [[buffer(8)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint blk = tg_pos.x;
    uint ks = tg_pos.y;
    uint base_col = blk << 5u;
    if (base_col >= p.padded_ff || ks >= max(p.k_splits, 1u)) return;
    if (base_col >= p.n_ff) {
        if (ks == 0u && tid < 32u) x_rot_out[base_col + tid] = 0.0f;
        return;
    }

    bool is_ec = (p.bits == 4u);
    bool fuse_norm = (p.fuse_norm != 0u);
    bool use_sh = fuse_norm || (p.padded_cols <= 2048u);
    threadgroup half2 lut64[64];
    threadgroup half x_sh[2048];
    threadgroup float2 tg_gu[32];
    vxt_init_lut64_sel(lut64, tid, is_ec);
    vxt_prepare_x_sh_256(
        fuse_norm, p.padded_cols, p.norm_dim, p.norm_eps,
        cur_in, norm_w, x_rot_in, x_sh, (threadgroup float*)tg_gu, tid, sg_id, lane
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint k_sp = max(p.k_splits, 1u);
    uint q_s = (steps + k_sp - 1u) / k_sp;
    uint s_beg = min(ks * q_s, steps);
    uint s_end = min(s_beg + q_s, steps);
    uint gh = lane >> 4u;
    uint tile_g = blk * 16u + sg_id;
    uint tile_u = tile_g + 8u;
    device const ushort* sc_g = scales_f16 + ulong(tile_g) * ulong(steps) * 8ul;
    device const ushort* sc_u = scales_f16 + ulong(tile_u) * ulong(steps) * 8ul;
    device const uint* pw_g = packed_words + ulong(tile_g) * ulong(steps) * 96ul;
    device const uint* pw_u = packed_words + ulong(tile_u) * ulong(steps) * 96ul;

    float4 acc_g = float4(0.0f);
    float4 acc_u = float4(0.0f);
    for (uint step = s_beg; step < s_end; ++step) {
        half4 xh0, xh1;
        vxt_load_xh_pair(use_sh, x_sh, x_rot_in, step, lane, xh0, xh1);
        device const uint* pg = pw_g + step * 96u;
        device const uint* pu = pw_u + step * 96u;
        uint soff = (step << 3u) + (gh << 2u);
        acc_g = vxt_eval_t3_lut(
            lut64, ((device const uint2*)pg)[lane], pg[64u + lane],
            *(device const half4*)(sc_g + soff), xh0, xh1, acc_g
        );
        acc_u = vxt_eval_t3_lut(
            lut64, ((device const uint2*)pu)[lane], pu[64u + lane],
            *(device const half4*)(sc_u + soff), xh0, xh1, acc_u
        );
    }

    vxt_ffn_gu_epilogue(
        blk, ks, simd_sum(acc_g) * 16.0f, simd_sum(acc_u) * 16.0f,
        tg_gu, x_rot_out, gu_part, sync_cnt, p, sg_id, lane
    );
}

// 7. High-Occupancy FP16 LUT + L1-Cached Split-K Batched `M = 2..4` GEMM for Turbo2, Turbo2.5 & Turbo3 (`256` threads)
[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_llm_turbo2_gemm4(
    device const ushort* scales_f16    [[buffer(0)]],
    device const uint* packed_words    [[buffer(1)]],
    device const half4* x_rot_h_batch  [[buffer(2)]],
    device float* y_batch              [[buffer(3)]],
    constant TurboGemmParams& p        [[buffer(4)]],
    device const half4* lut_bank       [[buffer(5)]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]],
    uint lane                          [[thread_index_in_simdgroup]],
    uint3 tg_pos                       [[threadgroup_position_in_grid]]
) {
    uint tps = p.tiles_per_sg & 255u;
    bool use_sk2 = (p.tiles_per_sg & 512u) != 0u;
    threadgroup half4 lut256[256];
    lut256[tid] = lut_bank[1024u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint p_h4 = p.padded_cols >> 2u;
    uint gh = lane >> 4u;

    if (use_sk2) {
        threadgroup float4 sk_part[16];
        uint sub = sg_id >> 1u;
        uint k_par = sg_id & 1u;
        uint tile = tg_pos.x * 4u + sub;
        uint h_s = (steps + 1u) >> 1u;
        uint kp_beg = min(k_par * h_s, steps);
        uint kp_end = min(kp_beg + h_s, steps);
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
        if ((tile << 2u) < p.rows) {
            device const ushort* sc_tile = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
            device const uint2* pw_tile = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);
            if (p.batch_m == 4u) {
                for (uint step = kp_beg; step < kp_end; ++step) {
                    uint2 w = pw_tile[(step << 5u) + lane];
                    half4 sc_h4 = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
                    uint x_h4 = (step << 6u) + (lane << 1u);
                    half4 c0a = lut256[w.x & 255u], c0b = lut256[(w.x >> 8u) & 255u];
                    half4 c1a = lut256[(w.x >> 16u) & 255u], c1b = lut256[w.x >> 24u];
                    half4 c2a = lut256[w.y & 255u], c2b = lut256[(w.y >> 8u) & 255u];
                    half4 c3a = lut256[(w.y >> 16u) & 255u], c3b = lut256[w.y >> 24u];
                    half4 xa = x_rot_h_batch[x_h4], xb = x_rot_h_batch[x_h4 + 1u];
                    acc0 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc0);
                    xa = x_rot_h_batch[p_h4 + x_h4]; xb = x_rot_h_batch[p_h4 + x_h4 + 1u];
                    acc1 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc1);
                    xa = x_rot_h_batch[2u * p_h4 + x_h4]; xb = x_rot_h_batch[2u * p_h4 + x_h4 + 1u];
                    acc2 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc2);
                    xa = x_rot_h_batch[3u * p_h4 + x_h4]; xb = x_rot_h_batch[3u * p_h4 + x_h4 + 1u];
                    acc3 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc3);
                }
            } else {
                for (uint step = kp_beg; step < kp_end; ++step) {
                    uint2 w = pw_tile[(step << 5u) + lane];
                    half4 sc_h4 = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
                    uint x_h4 = (step << 6u) + (lane << 1u);
                    half4 c0a = lut256[w.x & 255u], c0b = lut256[(w.x >> 8u) & 255u];
                    half4 c1a = lut256[(w.x >> 16u) & 255u], c1b = lut256[w.x >> 24u];
                    half4 c2a = lut256[w.y & 255u], c2b = lut256[(w.y >> 8u) & 255u];
                    half4 c3a = lut256[(w.y >> 16u) & 255u], c3b = lut256[w.y >> 24u];
                    half4 xa = x_rot_h_batch[x_h4], xb = x_rot_h_batch[x_h4 + 1u];
                    acc0 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc0);
                    if (p.batch_m > 1u) {
                        xa = x_rot_h_batch[p_h4 + x_h4]; xb = x_rot_h_batch[p_h4 + x_h4 + 1u];
                        acc1 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc1);
                    }
                    if (p.batch_m > 2u) {
                        xa = x_rot_h_batch[2u * p_h4 + x_h4]; xb = x_rot_h_batch[2u * p_h4 + x_h4 + 1u];
                        acc2 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc2);
                    }
                }
            }
        }
        float4 s0 = simd_sum(acc0) * 16.0f;
        float4 s1 = (p.batch_m > 1u) ? simd_sum(acc1) * 16.0f : float4(0.0f);
        float4 s2 = (p.batch_m > 2u) ? simd_sum(acc2) * 16.0f : float4(0.0f);
        float4 s3 = (p.batch_m > 3u) ? simd_sum(acc3) * 16.0f : float4(0.0f);
        if (k_par == 1u && lane == 0u) {
            sk_part[sub * 4u + 0u] = s0; sk_part[sub * 4u + 1u] = s1;
            sk_part[sub * 4u + 2u] = s2; sk_part[sub * 4u + 3u] = s3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
            vxt_store_tile4(y_batch, p.rows, tile, s0 + sk_part[sub * 4u + 0u]);
            if (p.batch_m > 1u) vxt_store_tile4(y_batch + p.rows, p.rows, tile, s1 + sk_part[sub * 4u + 1u]);
            if (p.batch_m > 2u) vxt_store_tile4(y_batch + 2u * p.rows, p.rows, tile, s2 + sk_part[sub * 4u + 2u]);
            if (p.batch_m > 3u) vxt_store_tile4(y_batch + 3u * p.rows, p.rows, tile, s3 + sk_part[sub * 4u + 3u]);
        }
        return;
    }

    uint sg_base = (tg_pos.x * 8u + sg_id) * tps;
    for (uint kt = 0u; kt < tps; ++kt) {
        uint tile = sg_base + kt;
        if ((tile << 2u) >= p.rows) break;
        device const ushort* sc_tile = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint2* pw_tile = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            uint2 w = pw_tile[(step << 5u) + lane];
            half4 sc_h4 = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
            uint x_h4 = (step << 6u) + (lane << 1u);
            acc0 = vxt_eval_t2_lut(lut256, w, sc_h4, x_rot_h_batch[x_h4], x_rot_h_batch[x_h4 + 1u], acc0);
            if (p.batch_m > 1u) acc1 = vxt_eval_t2_lut(lut256, w, sc_h4, x_rot_h_batch[p_h4 + x_h4], x_rot_h_batch[p_h4 + x_h4 + 1u], acc1);
            if (p.batch_m > 2u) acc2 = vxt_eval_t2_lut(lut256, w, sc_h4, x_rot_h_batch[2u * p_h4 + x_h4], x_rot_h_batch[2u * p_h4 + x_h4 + 1u], acc2);
            if (p.batch_m > 3u) acc3 = vxt_eval_t2_lut(lut256, w, sc_h4, x_rot_h_batch[3u * p_h4 + x_h4], x_rot_h_batch[3u * p_h4 + x_h4 + 1u], acc3);
        }
        float4 s0 = simd_sum(acc0) * 16.0f;
        float4 s1 = (p.batch_m > 1u) ? simd_sum(acc1) * 16.0f : float4(0.0f);
        float4 s2 = (p.batch_m > 2u) ? simd_sum(acc2) * 16.0f : float4(0.0f);
        float4 s3 = (p.batch_m > 3u) ? simd_sum(acc3) * 16.0f : float4(0.0f);
        if (lane == 0u) {
            vxt_store_tile4(y_batch, p.rows, tile, s0);
            if (p.batch_m > 1u) vxt_store_tile4(y_batch + p.rows, p.rows, tile, s1);
            if (p.batch_m > 2u) vxt_store_tile4(y_batch + 2u * p.rows, p.rows, tile, s2);
            if (p.batch_m > 3u) vxt_store_tile4(y_batch + 3u * p.rows, p.rows, tile, s3);
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_llm_turbo25_gemm4(
    device const ushort* scales_f16    [[buffer(0)]],
    device const uint* packed_words    [[buffer(1)]],
    device const half4* x_rot_h_batch  [[buffer(2)]],
    device float* y_batch              [[buffer(3)]],
    constant TurboGemmParams& p        [[buffer(4)]],
    device const half4* lut_bank       [[buffer(5)]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]],
    uint lane                          [[thread_index_in_simdgroup]],
    uint3 tg_pos                       [[threadgroup_position_in_grid]]
) {
    uint tps = p.tiles_per_sg & 255u;
    bool use_sk2 = (p.tiles_per_sg & 512u) != 0u;
    threadgroup half4 lut1k[1024];
    for (uint i = tid; i < 1024u; i += 256u) lut1k[i] = lut_bank[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint p_h4 = p.padded_cols >> 2u;
    uint gh = lane >> 4u;

    if (use_sk2) {
        threadgroup float4 sk_part[16];
        uint sub = sg_id >> 1u;
        uint k_par = sg_id & 1u;
        uint tile = tg_pos.x * 4u + sub;
        uint h_s = (steps + 1u) >> 1u;
        uint kp_beg = min(k_par * h_s, steps);
        uint kp_end = min(kp_beg + h_s, steps);
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
        if ((tile << 2u) < p.rows) {
            device const ushort* sc_tile = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
            device const uint* pw_tile = packed_words + ulong(tile) * ulong(steps) * 80ul;
            if (p.batch_m == 4u) {
                for (uint step = kp_beg; step < kp_end; ++step) {
                    device const uint* p80 = pw_tile + step * 80u;
                    uint2 w01 = ((device const uint2*)p80)[lane];
                    uint h16 = uint(((device const ushort*)(p80 + 64u))[lane]);
                    half4 sc_h4 = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
                    uint x_h4 = (step << 6u) + (lane << 1u);
                    half4 c0a = lut1k[w01.x & 1023u], c0b = lut1k[(w01.x >> 10u) & 1023u];
                    half4 c1a = lut1k[(w01.x >> 20u) & 1023u], c1b = lut1k[w01.y & 1023u];
                    half4 c2a = lut1k[(w01.y >> 10u) & 1023u], c2b = lut1k[(w01.y >> 20u) & 1023u];
                    half4 c3a = lut1k[(h16 & 0xffu) | ((w01.x >> 30u) << 8u)], c3b = lut1k[(h16 >> 8u) | ((w01.y >> 30u) << 8u)];
                    half4 xa = x_rot_h_batch[x_h4], xb = x_rot_h_batch[x_h4 + 1u];
                    acc0 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc0);
                    xa = x_rot_h_batch[p_h4 + x_h4]; xb = x_rot_h_batch[p_h4 + x_h4 + 1u];
                    acc1 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc1);
                    xa = x_rot_h_batch[2u * p_h4 + x_h4]; xb = x_rot_h_batch[2u * p_h4 + x_h4 + 1u];
                    acc2 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc2);
                    xa = x_rot_h_batch[3u * p_h4 + x_h4]; xb = x_rot_h_batch[3u * p_h4 + x_h4 + 1u];
                    acc3 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc3);
                }
            } else {
                for (uint step = kp_beg; step < kp_end; ++step) {
                    device const uint* p80 = pw_tile + step * 80u;
                    uint2 w01 = ((device const uint2*)p80)[lane];
                    uint h16 = uint(((device const ushort*)(p80 + 64u))[lane]);
                    half4 sc_h4 = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
                    uint x_h4 = (step << 6u) + (lane << 1u);
                    half4 c0a = lut1k[w01.x & 1023u], c0b = lut1k[(w01.x >> 10u) & 1023u];
                    half4 c1a = lut1k[(w01.x >> 20u) & 1023u], c1b = lut1k[w01.y & 1023u];
                    half4 c2a = lut1k[(w01.y >> 10u) & 1023u], c2b = lut1k[(w01.y >> 20u) & 1023u];
                    half4 c3a = lut1k[(h16 & 0xffu) | ((w01.x >> 30u) << 8u)], c3b = lut1k[(h16 >> 8u) | ((w01.y >> 30u) << 8u)];
                    half4 xa = x_rot_h_batch[x_h4], xb = x_rot_h_batch[x_h4 + 1u];
                    acc0 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc0);
                    if (p.batch_m > 1u) {
                        xa = x_rot_h_batch[p_h4 + x_h4]; xb = x_rot_h_batch[p_h4 + x_h4 + 1u];
                        acc1 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc1);
                    }
                    if (p.batch_m > 2u) {
                        xa = x_rot_h_batch[2u * p_h4 + x_h4]; xb = x_rot_h_batch[2u * p_h4 + x_h4 + 1u];
                        acc2 = fma(float4(sc_h4), float4(half4(vxt_dot2_h4(c0a, c0b, xa, xb), vxt_dot2_h4(c1a, c1b, xa, xb), vxt_dot2_h4(c2a, c2b, xa, xb), vxt_dot2_h4(c3a, c3b, xa, xb))), acc2);
                    }
                }
            }
        }
        float4 s0 = simd_sum(acc0) * 16.0f;
        float4 s1 = (p.batch_m > 1u) ? simd_sum(acc1) * 16.0f : float4(0.0f);
        float4 s2 = (p.batch_m > 2u) ? simd_sum(acc2) * 16.0f : float4(0.0f);
        float4 s3 = (p.batch_m > 3u) ? simd_sum(acc3) * 16.0f : float4(0.0f);
        if (k_par == 1u && lane == 0u) {
            sk_part[sub * 4u + 0u] = s0; sk_part[sub * 4u + 1u] = s1;
            sk_part[sub * 4u + 2u] = s2; sk_part[sub * 4u + 3u] = s3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
            vxt_store_tile4(y_batch, p.rows, tile, s0 + sk_part[sub * 4u + 0u]);
            if (p.batch_m > 1u) vxt_store_tile4(y_batch + p.rows, p.rows, tile, s1 + sk_part[sub * 4u + 1u]);
            if (p.batch_m > 2u) vxt_store_tile4(y_batch + 2u * p.rows, p.rows, tile, s2 + sk_part[sub * 4u + 2u]);
            if (p.batch_m > 3u) vxt_store_tile4(y_batch + 3u * p.rows, p.rows, tile, s3 + sk_part[sub * 4u + 3u]);
        }
        return;
    }

    uint sg_base = (tg_pos.x * 8u + sg_id) * tps;
    for (uint kt = 0u; kt < tps; ++kt) {
        uint tile = sg_base + kt;
        if ((tile << 2u) >= p.rows) break;
        device const ushort* sc_tile = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw_tile = packed_words + ulong(tile) * ulong(steps) * 80ul;
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const uint* p80 = pw_tile + step * 80u;
            uint2 w01 = ((device const uint2*)p80)[lane];
            uint h16 = uint(((device const ushort*)(p80 + 64u))[lane]);
            half4 sc_h4 = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
            uint x_h4 = (step << 6u) + (lane << 1u);
            acc0 = vxt_eval_t25_lut1k(lut1k, w01, h16, sc_h4, x_rot_h_batch[x_h4], x_rot_h_batch[x_h4 + 1u], acc0);
            if (p.batch_m > 1u) acc1 = vxt_eval_t25_lut1k(lut1k, w01, h16, sc_h4, x_rot_h_batch[p_h4 + x_h4], x_rot_h_batch[p_h4 + x_h4 + 1u], acc1);
            if (p.batch_m > 2u) acc2 = vxt_eval_t25_lut1k(lut1k, w01, h16, sc_h4, x_rot_h_batch[2u * p_h4 + x_h4], x_rot_h_batch[2u * p_h4 + x_h4 + 1u], acc2);
            if (p.batch_m > 3u) acc3 = vxt_eval_t25_lut1k(lut1k, w01, h16, sc_h4, x_rot_h_batch[3u * p_h4 + x_h4], x_rot_h_batch[3u * p_h4 + x_h4 + 1u], acc3);
        }
        float4 s0 = simd_sum(acc0) * 16.0f;
        float4 s1 = (p.batch_m > 1u) ? simd_sum(acc1) * 16.0f : float4(0.0f);
        float4 s2 = (p.batch_m > 2u) ? simd_sum(acc2) * 16.0f : float4(0.0f);
        float4 s3 = (p.batch_m > 3u) ? simd_sum(acc3) * 16.0f : float4(0.0f);
        if (lane == 0u) {
            vxt_store_tile4(y_batch, p.rows, tile, s0);
            if (p.batch_m > 1u) vxt_store_tile4(y_batch + p.rows, p.rows, tile, s1);
            if (p.batch_m > 2u) vxt_store_tile4(y_batch + 2u * p.rows, p.rows, tile, s2);
            if (p.batch_m > 3u) vxt_store_tile4(y_batch + 3u * p.rows, p.rows, tile, s3);
        }
    }
}

inline half vxt_dot_u4_h2(half2 u0, half2 u1, half2 u2, half2 u3, half4 xa, half4 xb) {
    half2 p = u0 * xa.xy;
    p = fma(u1, xa.zw, p);
    p = fma(u2, xb.xy, p);
    p = fma(u3, xb.zw, p);
    return p.x + p.y;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_llm_turbo3_gemm4(
    device const ushort* scales_f16    [[buffer(0)]],
    device const uint* packed_words    [[buffer(1)]],
    device const half4* x_rot_h_batch  [[buffer(2)]],
    device float* y_batch              [[buffer(3)]],
    constant TurboGemmParams& p        [[buffer(4)]],
    device const half4* lut_bank       [[buffer(5)]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]],
    uint lane                          [[thread_index_in_simdgroup]],
    uint3 tg_pos                       [[threadgroup_position_in_grid]]
) {
    bool is_ec = (p.tiles_per_sg & 256u) != 0u;
    bool use_sk2 = (p.tiles_per_sg & 512u) != 0u;
    uint tps = p.tiles_per_sg & 255u;
    threadgroup half2 lut64[64];
    if (tid < 64u) lut64[tid] = ((device const half2*)(lut_bank + (is_ec ? 1312u : 1280u)))[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint p_h4 = p.padded_cols >> 2u;
    uint gh = lane >> 4u;

    if (use_sk2) {
        threadgroup float4 sk_part[16];
        uint sub = sg_id >> 1u;
        uint k_par = sg_id & 1u;
        uint tile = tg_pos.x * 4u + sub;
        uint h_s = (steps + 1u) >> 1u;
        uint kp_beg = min(k_par * h_s, steps);
        uint kp_end = min(kp_beg + h_s, steps);
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
        if ((tile << 2u) < p.rows) {
            device const ushort* sc_tile = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
            device const uint* pw_tile = packed_words + ulong(tile) * ulong(steps) * 96ul;
            if (p.batch_m == 4u) {
                for (uint step = kp_beg; step < kp_end; ++step) {
                    uint x_h4 = (step << 6u) + (lane << 1u);
                    half4 x0a = x_rot_h_batch[x_h4], x0b = x_rot_h_batch[x_h4 + 1u];
                    half4 x1a = x_rot_h_batch[p_h4 + x_h4], x1b = x_rot_h_batch[p_h4 + x_h4 + 1u];
                    half4 x2a = x_rot_h_batch[2u * p_h4 + x_h4], x2b = x_rot_h_batch[2u * p_h4 + x_h4 + 1u];
                    half4 x3a = x_rot_h_batch[3u * p_h4 + x_h4], x3b = x_rot_h_batch[3u * p_h4 + x_h4 + 1u];

                    device const uint* p96 = pw_tile + step * 96u;
                    uint2 rw01 = ((device const uint2*)p96)[lane];
                    uint rw2 = p96[64u + lane];
                    float4 sf = float4(*(device const half4*)(sc_tile + (step << 3u) + (gh << 2u)));

                    half4 d_m0, d_m1, d_m2, d_m3;
                    half2 u0 = lut64[rw01.x & 63u], u1 = lut64[(rw01.x >> 6u) & 63u];
                    half2 u2 = lut64[(rw01.x >> 12u) & 63u], u3 = lut64[(rw01.x >> 18u) & 63u];
                    d_m0.x = vxt_dot_u4_h2(u0, u1, u2, u3, x0a, x0b);
                    d_m1.x = vxt_dot_u4_h2(u0, u1, u2, u3, x1a, x1b);
                    d_m2.x = vxt_dot_u4_h2(u0, u1, u2, u3, x2a, x2b);
                    d_m3.x = vxt_dot_u4_h2(u0, u1, u2, u3, x3a, x3b);

                    u0 = lut64[rw01.y & 63u]; u1 = lut64[(rw01.y >> 6u) & 63u];
                    u2 = lut64[(rw01.y >> 12u) & 63u]; u3 = lut64[(rw01.y >> 18u) & 63u];
                    d_m0.y = vxt_dot_u4_h2(u0, u1, u2, u3, x0a, x0b);
                    d_m1.y = vxt_dot_u4_h2(u0, u1, u2, u3, x1a, x1b);
                    d_m2.y = vxt_dot_u4_h2(u0, u1, u2, u3, x2a, x2b);
                    d_m3.y = vxt_dot_u4_h2(u0, u1, u2, u3, x3a, x3b);

                    u0 = lut64[rw2 & 63u]; u1 = lut64[(rw2 >> 6u) & 63u];
                    u2 = lut64[(rw2 >> 12u) & 63u]; u3 = lut64[(rw2 >> 18u) & 63u];
                    d_m0.z = vxt_dot_u4_h2(u0, u1, u2, u3, x0a, x0b);
                    d_m1.z = vxt_dot_u4_h2(u0, u1, u2, u3, x1a, x1b);
                    d_m2.z = vxt_dot_u4_h2(u0, u1, u2, u3, x2a, x2b);
                    d_m3.z = vxt_dot_u4_h2(u0, u1, u2, u3, x3a, x3b);

                    uint c33 = (rw01.x >> 30u) | ((rw01.y >> 28u) & 0x0Cu) | ((rw2 >> 26u) & 0x30u);
                    u0 = lut64[(rw01.x >> 24u) & 63u]; u1 = lut64[(rw01.y >> 24u) & 63u];
                    u2 = lut64[(rw2 >> 24u) & 63u]; u3 = lut64[c33];
                    d_m0.w = vxt_dot_u4_h2(u0, u1, u2, u3, x0a, x0b);
                    d_m1.w = vxt_dot_u4_h2(u0, u1, u2, u3, x1a, x1b);
                    d_m2.w = vxt_dot_u4_h2(u0, u1, u2, u3, x2a, x2b);
                    d_m3.w = vxt_dot_u4_h2(u0, u1, u2, u3, x3a, x3b);

                    acc0 = fma(sf, float4(d_m0), acc0);
                    acc1 = fma(sf, float4(d_m1), acc1);
                    acc2 = fma(sf, float4(d_m2), acc2);
                    acc3 = fma(sf, float4(d_m3), acc3);
                }
            } else {
                for (uint step = kp_beg; step < kp_end; ++step) {
                    uint x_h4 = (step << 6u) + (lane << 1u);
                    half4 x0a = x_rot_h_batch[x_h4], x0b = x_rot_h_batch[x_h4 + 1u];
                    half4 x1a = half4(0.0h), x1b = half4(0.0h);
                    half4 x2a = half4(0.0h), x2b = half4(0.0h);
                    if (p.batch_m > 1u) { x1a = x_rot_h_batch[p_h4 + x_h4]; x1b = x_rot_h_batch[p_h4 + x_h4 + 1u]; }
                    if (p.batch_m > 2u) { x2a = x_rot_h_batch[2u * p_h4 + x_h4]; x2b = x_rot_h_batch[2u * p_h4 + x_h4 + 1u]; }

                    device const uint* p96 = pw_tile + step * 96u;
                    uint2 rw01 = ((device const uint2*)p96)[lane];
                    uint rw2 = p96[64u + lane];
                    float4 sf = float4(*(device const half4*)(sc_tile + (step << 3u) + (gh << 2u)));

                    half4 d_m0, d_m1, d_m2;
                    half2 u0 = lut64[rw01.x & 63u], u1 = lut64[(rw01.x >> 6u) & 63u];
                    half2 u2 = lut64[(rw01.x >> 12u) & 63u], u3 = lut64[(rw01.x >> 18u) & 63u];
                    d_m0.x = vxt_dot_u4_h2(u0, u1, u2, u3, x0a, x0b);
                    if (p.batch_m > 1u) d_m1.x = vxt_dot_u4_h2(u0, u1, u2, u3, x1a, x1b);
                    if (p.batch_m > 2u) d_m2.x = vxt_dot_u4_h2(u0, u1, u2, u3, x2a, x2b);

                    u0 = lut64[rw01.y & 63u]; u1 = lut64[(rw01.y >> 6u) & 63u];
                    u2 = lut64[(rw01.y >> 12u) & 63u]; u3 = lut64[(rw01.y >> 18u) & 63u];
                    d_m0.y = vxt_dot_u4_h2(u0, u1, u2, u3, x0a, x0b);
                    if (p.batch_m > 1u) d_m1.y = vxt_dot_u4_h2(u0, u1, u2, u3, x1a, x1b);
                    if (p.batch_m > 2u) d_m2.y = vxt_dot_u4_h2(u0, u1, u2, u3, x2a, x2b);

                    u0 = lut64[rw2 & 63u]; u1 = lut64[(rw2 >> 6u) & 63u];
                    u2 = lut64[(rw2 >> 12u) & 63u]; u3 = lut64[(rw2 >> 18u) & 63u];
                    d_m0.z = vxt_dot_u4_h2(u0, u1, u2, u3, x0a, x0b);
                    if (p.batch_m > 1u) d_m1.z = vxt_dot_u4_h2(u0, u1, u2, u3, x1a, x1b);
                    if (p.batch_m > 2u) d_m2.z = vxt_dot_u4_h2(u0, u1, u2, u3, x2a, x2b);

                    uint c33 = (rw01.x >> 30u) | ((rw01.y >> 28u) & 0x0Cu) | ((rw2 >> 26u) & 0x30u);
                    u0 = lut64[(rw01.x >> 24u) & 63u]; u1 = lut64[(rw01.y >> 24u) & 63u];
                    u2 = lut64[(rw2 >> 24u) & 63u]; u3 = lut64[c33];
                    d_m0.w = vxt_dot_u4_h2(u0, u1, u2, u3, x0a, x0b);
                    if (p.batch_m > 1u) d_m1.w = vxt_dot_u4_h2(u0, u1, u2, u3, x1a, x1b);
                    if (p.batch_m > 2u) d_m2.w = vxt_dot_u4_h2(u0, u1, u2, u3, x2a, x2b);

                    acc0 = fma(sf, float4(d_m0), acc0);
                    if (p.batch_m > 1u) acc1 = fma(sf, float4(d_m1), acc1);
                    if (p.batch_m > 2u) acc2 = fma(sf, float4(d_m2), acc2);
                }
            }
        }
        float4 s0 = simd_sum(acc0) * 16.0f;
        float4 s1 = (p.batch_m > 1u) ? simd_sum(acc1) * 16.0f : float4(0.0f);
        float4 s2 = (p.batch_m > 2u) ? simd_sum(acc2) * 16.0f : float4(0.0f);
        float4 s3 = (p.batch_m > 3u) ? simd_sum(acc3) * 16.0f : float4(0.0f);
        if (k_par == 1u && lane == 0u) {
            sk_part[sub * 4u + 0u] = s0; sk_part[sub * 4u + 1u] = s1;
            sk_part[sub * 4u + 2u] = s2; sk_part[sub * 4u + 3u] = s3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k_par == 0u && lane == 0u && (tile << 2u) < p.rows) {
            vxt_store_tile4(y_batch, p.rows, tile, s0 + sk_part[sub * 4u + 0u]);
            if (p.batch_m > 1u) vxt_store_tile4(y_batch + p.rows, p.rows, tile, s1 + sk_part[sub * 4u + 1u]);
            if (p.batch_m > 2u) vxt_store_tile4(y_batch + 2u * p.rows, p.rows, tile, s2 + sk_part[sub * 4u + 2u]);
            if (p.batch_m > 3u) vxt_store_tile4(y_batch + 3u * p.rows, p.rows, tile, s3 + sk_part[sub * 4u + 3u]);
        }
        return;
    }

    uint sg_base = (tg_pos.x * 8u + sg_id) * tps;
    for (uint kt = 0u; kt < tps; ++kt) {
        uint tile = sg_base + kt;
        if ((tile << 2u) >= p.rows) break;
        device const ushort* sc_tile = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw_tile = packed_words + ulong(tile) * ulong(steps) * 96ul;
        float4 acc0 = float4(0.0f), acc1 = float4(0.0f), acc2 = float4(0.0f), acc3 = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const uint* p96 = pw_tile + step * 96u;
            uint2 rw01 = ((device const uint2*)p96)[lane];
            uint rw2 = p96[64u + lane];
            half4 sc_h4 = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
            uint x_h4 = (step << 6u) + (lane << 1u);
            acc0 = vxt_eval_t3_lut(lut64, rw01, rw2, sc_h4, x_rot_h_batch[x_h4], x_rot_h_batch[x_h4 + 1u], acc0);
            if (p.batch_m > 1u) acc1 = vxt_eval_t3_lut(lut64, rw01, rw2, sc_h4, x_rot_h_batch[p_h4 + x_h4], x_rot_h_batch[p_h4 + x_h4 + 1u], acc1);
            if (p.batch_m > 2u) acc2 = vxt_eval_t3_lut(lut64, rw01, rw2, sc_h4, x_rot_h_batch[2u * p_h4 + x_h4], x_rot_h_batch[2u * p_h4 + x_h4 + 1u], acc2);
            if (p.batch_m > 3u) acc3 = vxt_eval_t3_lut(lut64, rw01, rw2, sc_h4, x_rot_h_batch[3u * p_h4 + x_h4], x_rot_h_batch[3u * p_h4 + x_h4 + 1u], acc3);
        }
        float4 s0 = simd_sum(acc0) * 16.0f;
        float4 s1 = (p.batch_m > 1u) ? simd_sum(acc1) * 16.0f : float4(0.0f);
        float4 s2 = (p.batch_m > 2u) ? simd_sum(acc2) * 16.0f : float4(0.0f);
        float4 s3 = (p.batch_m > 3u) ? simd_sum(acc3) * 16.0f : float4(0.0f);
        if (lane == 0u) {
            vxt_store_tile4(y_batch, p.rows, tile, s0);
            if (p.batch_m > 1u) vxt_store_tile4(y_batch + p.rows, p.rows, tile, s1);
            if (p.batch_m > 2u) vxt_store_tile4(y_batch + 2u * p.rows, p.rows, tile, s2);
            if (p.batch_m > 3u) vxt_store_tile4(y_batch + 3u * p.rows, p.rows, tile, s3);
        }
    }
}

// 8. Batched Causal Conv1D + DeltaNet (plus Ultra-Fused 1-to-1 Single-Kernel Conv1D+DeltaNet+FWHT32!)
kernel void vxt_deltanet_conv_batch(
    device const float* qkv_gate        [[buffer(0)]],
    device const float* ssm_conv1d      [[buffer(1)]],
    device float* conv_state            [[buffer(2)]],
    device float* conv_bak              [[buffer(3)]],
    device float* conv_silu_out         [[buffer(4)]],
    constant TurboDeltaConvParams& p    [[buffer(5)]],
    uint c                              [[thread_position_in_grid]]
) {
    if (c >= p.conv_ch) return;
    uint k_sz = p.k_sz;
    uint hist = k_sz - 1u;
    device const float* w = ssm_conv1d + c * k_sz;

    float h0 = (hist > 0u) ? conv_state[c] : 0.0f;
    float h1 = (hist > 1u) ? conv_state[p.conv_ch + c] : 0.0f;
    float h2 = (hist > 2u) ? conv_state[2u * p.conv_ch + c] : 0.0f;

    for (uint m = 0u; m < p.batch_m; ++m) {
        float x_c = qkv_gate[ulong(m) * ulong(p.qkv_gate_stride) + ulong(c)];
        float acc = 0.0f;
        if (hist == 3u) {
            acc = fma(x_c, w[3], fma(h2, w[2], fma(h1, w[1], h0 * w[0])));
            h0 = h1; h1 = h2; h2 = x_c;
            if (p.save_backup != 0u && m == 0u) {
                conv_bak[c] = h0;
                conv_bak[p.conv_ch + c] = h1;
                conv_bak[2u * p.conv_ch + c] = h2;
            }
        } else {
            for (uint t = 0u; t < hist; ++t) {
                float hv = (t == 0u) ? h0 : ((t == 1u) ? h1 : ((t == 2u) ? h2 : conv_state[t * p.conv_ch + c]));
                acc = fma(hv, w[t], acc);
            }
            acc = fma(x_c, w[hist], acc);
            if (hist > 0u) {
                for (uint t = 0u; t + 1u < hist; ++t) {
                    conv_state[t * p.conv_ch + c] = conv_state[(t + 1u) * p.conv_ch + c];
                }
                conv_state[(hist - 1u) * p.conv_ch + c] = x_c;
                h0 = conv_state[c];
                if (hist > 1u) h1 = conv_state[p.conv_ch + c];
                if (hist > 2u) h2 = conv_state[2u * p.conv_ch + c];
            }
            if (p.save_backup != 0u && m == 0u) {
                for (uint t = 0u; t < hist; ++t) {
                    conv_bak[t * p.conv_ch + c] = conv_state[t * p.conv_ch + c];
                }
            }
        }
        conv_silu_out[ulong(m) * ulong(p.conv_ch) + ulong(c)] = acc / (1.0f + exp(-acc));
    }

    if (hist == 3u) {
        conv_state[c] = h0;
        conv_state[p.conv_ch + c] = h1;
        conv_state[2u * p.conv_ch + c] = h2;
    }
}

// 8b. Full-Occupancy 64-TG `M = 1` DeltaNet (`prep16` + `state64` + `out2` for `dk=128, dv=128`)
kernel void vxt_dnet_m1_prep16(
    device const float* qkv_in          [[buffer(0)]],
    device const float* ssm_conv1d      [[buffer(1)]],
    device const float* ssm_a           [[buffer(2)]],
    device const float* ssm_dt_bias     [[buffer(3)]],
    device float* conv_state            [[buffer(4)]],
    device const half* ab_rot_f16       [[buffer(5)]],
    device const float* x_rot_in        [[buffer(6)]],
    device float* dnet_tmp              [[buffer(7)]],
    constant TurboDeltaHeadsParams& p   [[buffer(8)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    if (h >= p.n_vh) return;

    uint qk_dim = p.n_vh * 128u;
    uint conv_ch = 3u * qk_dim;
    device float* q_all = dnet_tmp;
    device float* k_all = dnet_tmp + qk_dim;
    device float* v_all = dnet_tmp + 2u * qk_dim;
    device float2* db_all = (device float2*)(dnet_tmp + 4u * qk_dim);

    threadgroup float q_sh[128];
    threadgroup float k_sh[128];
    threadgroup float sg_ab[4];

    for (uint sub = 0u; sub < 3u; ++sub) {
        uint c = sub * qk_dim + h * 128u + tid;
        device const float* w = ssm_conv1d + c * 4u;
        float h0 = conv_state[c];
        float h1 = conv_state[conv_ch + c];
        float h2 = conv_state[2u * conv_ch + c];
        float x_c = qkv_in[c];
        float acc = fma(x_c, w[3], fma(h2, w[2], fma(h1, w[1], h0 * w[0])));
        conv_state[c] = h1;
        conv_state[conv_ch + c] = h2;
        conv_state[2u * conv_ch + c] = x_c;
        float sv = acc / (1.0f + exp(-acc));
        if (sub == 0u) q_sh[tid] = sv;
        else if (sub == 1u) k_sh[tid] = sv;
        else v_all[h * 128u + tid] = sv;
    }

    uint n4 = p.padded_embd >> 2u;
    uint ab_row = (sg_id < 2u) ? h : (p.n_vh + h);
    uint ab_sub = sg_id & 1u;
    device const half4* wr_ab4 = (device const half4*)(ab_rot_f16 + ulong(ab_row) * ulong(p.padded_embd));
    device const float4* xr4 = (device const float4*)x_rot_in;
    float acc_ab = 0.0f;
    for (uint c4 = ab_sub * 32u + lane; c4 < n4; c4 += 64u) {
        acc_ab += dot(float4(wr_ab4[c4]), xr4[c4]);
    }
    acc_ab = simd_sum(acc_ab);
    if (lane == 0u) sg_ab[sg_id] = acc_ab;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float4 qv = ((threadgroup const float4*)q_sh)[lane];
        float sc = 0.08838834764831845f * rsqrt(simd_sum(dot(qv, qv)) + p.eps);
        ((device float4*)(q_all + h * 128u))[lane] = qv * sc;
    } else if (sg_id == 1u) {
        float4 kv = ((threadgroup const float4*)k_sh)[lane];
        float sc = rsqrt(simd_sum(dot(kv, kv)) + p.eps);
        ((device float4*)(k_all + h * 128u))[lane] = kv * sc;
    } else if (sg_id == 2u && lane == 0u) {
        float alpha_val = sg_ab[0] + sg_ab[1];
        float beta_val = sg_ab[2] + sg_ab[3];
        float beta = 1.0f / (1.0f + exp(-beta_val));
        float x_sp = alpha_val + ssm_dt_bias[h];
        float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? exp(x_sp) : log(1.0f + exp(x_sp)));
        db_all[h] = float2(exp(sp * ssm_a[h]), beta);
    }
}

kernel void vxt_dnet_m1_state64(
    VXT_COHERENT_DEV float* dnet_tmp    [[buffer(0)]],
    device half* ssm_state              [[buffer(1)]],
    constant TurboDeltaHeadsParams& p   [[buffer(2)]],
    device const float* z_in            [[buffer(3)]],
    device const float* ssm_norm        [[buffer(4)]],
    device half* x_rot_h_out            [[buffer(5)]],
    device float* x_rot_f_out           [[buffer(6)]],
    device atomic_uint* sync_cnt        [[buffer(7)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint tg_id = tg_pos.x;
    uint h = tg_id >> 2u;
    uint sub = tg_id & 3u;
    if (h >= p.n_vh) return;

    uint qk_dim = p.n_vh * 128u;
    float4 q4 = ((VXT_COHERENT_DEV const float4*)(dnet_tmp + h * 128u))[lane];
    float4 k4 = ((VXT_COHERENT_DEV const float4*)(dnet_tmp + qk_dim + h * 128u))[lane];
    VXT_COHERENT_DEV const float* v_h = dnet_tmp + 2u * qk_dim + h * 128u;
    VXT_COHERENT_DEV float* o_h = dnet_tmp + 3u * qk_dim + h * 128u;
    float2 db = ((VXT_COHERENT_DEV const float2*)(dnet_tmp + 4u * qk_dim))[h];
    float decay = db.x;
    float beta = db.y;

    device half* sh_base = ssm_state + ulong(h) * 16384ul;
    uint base_jv = (sub << 5u) + sg_id;

    for (uint g = 0u; g < 2u; ++g) {
        uint jv0 = base_jv + (g << 4u);
        uint jv1 = jv0 + 4u;
        uint jv2 = jv0 + 8u;
        uint jv3 = jv0 + 12u;
        device half4* row0 = (device half4*)(sh_base + (jv0 << 7u));
        device half4* row1 = (device half4*)(sh_base + (jv1 << 7u));
        device half4* row2 = (device half4*)(sh_base + (jv2 << 7u));
        device half4* row3 = (device half4*)(sh_base + (jv3 << 7u));
        float4 r0 = float4(row0[lane]) * decay;
        float4 r1 = float4(row1[lane]) * decay;
        float4 r2 = float4(row2[lane]) * decay;
        float4 r3 = float4(row3[lane]) * decay;
        float4 sk4 = simd_sum(float4(dot(r0, k4), dot(r1, k4), dot(r2, k4), dot(r3, k4)));
        float4 d4 = (float4(v_h[jv0], v_h[jv1], v_h[jv2], v_h[jv3]) - sk4) * beta;
        r0 = fma(float4(d4.x), k4, r0);
        r1 = fma(float4(d4.y), k4, r1);
        r2 = fma(float4(d4.z), k4, r2);
        r3 = fma(float4(d4.w), k4, r3);
        row0[lane] = half4(r0);
        row1[lane] = half4(r1);
        row2[lane] = half4(r2);
        row3[lane] = half4(r3);
        float4 o4 = simd_sum(float4(dot(r0, q4), dot(r1, q4), dot(r2, q4), dot(r3, q4)));
        if (lane == 0u) {
            o_h[jv0] = o4.x;
            o_h[jv1] = o4.y;
            o_h[jv2] = o4.z;
            o_h[jv3] = o4.w;
        }
    }

    if (p.fuse_out != 0u) {
        if (tg_id == 0u) {
            uint inner_dim = p.n_vh * 128u;
            for (uint col = inner_dim + tid; col < p.padded_inner; col += 128u) {
                x_rot_h_out[col] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        threadgroup uint head_done[1];
        if (tid == 0u) {
            vxt_device_fence();
            uint prev = atomic_fetch_add_explicit(&sync_cnt[p.sync_slot + h], 1u, memory_order_relaxed);
            if (prev == 3u) {
                atomic_store_explicit(&sync_cnt[p.sync_slot + h], 0u, memory_order_relaxed);
                head_done[0] = 1u;
            } else {
                head_done[0] = 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (head_done[0] != 0u && sg_id == 0u) {
            vxt_device_fence();
            float o0 = o_h[lane];
            float o1 = o_h[32u + lane];
            float o2 = o_h[64u + lane];
            float o3 = o_h[96u + lane];
            float sq_o = simd_sum(fma(o3, o3, fma(o2, o2, fma(o1, o1, o0 * o0))));
            float o_scale = rsqrt(sq_o * 0.0078125f + p.eps);
            device const float* z_h = z_in + h * 128u;
            float o_arr[4] = { o0, o1, o2, o3 };
            for (uint b = 0u; b < 4u; ++b) {
                uint jv = (b << 5u) + lane;
                float zv = z_h[jv];
                float out_v = (o_arr[b] * o_scale * ssm_norm[jv]) * (zv / (1.0f + exp(-zv)));
                uint col = h * 128u + jv;
                float rot = vxt_fwht32_lane(vxt_apply_col_sign(out_v, col), lane);
                x_rot_h_out[col] = half(rot * 0.0625f);
            }
        }
    }
    (void)x_rot_f_out;
}

kernel void vxt_dnet_m1_out2(
    device const float* dnet_tmp        [[buffer(0)]],
    device const float* z_in            [[buffer(1)]],
    device const float* ssm_norm        [[buffer(2)]],
    device float* x_rot_out             [[buffer(3)]],
    constant TurboDeltaHeadsParams& p   [[buffer(4)]],
    device half* x_rot_h_out            [[buffer(5)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x * 8u + sg_id;
    uint qk_dim = p.n_vh * 128u;
    if (h < p.n_vh) {
        device const float* o_h = dnet_tmp + 3u * qk_dim + h * 128u;
        float o0 = o_h[lane];
        float o1 = o_h[32u + lane];
        float o2 = o_h[64u + lane];
        float o3 = o_h[96u + lane];
        float sq_o = simd_sum(fma(o3, o3, fma(o2, o2, fma(o1, o1, o0 * o0))));
        float o_scale = rsqrt(sq_o * 0.0078125f + p.eps);
        device const float* z_h = z_in + h * 128u;
        float o_arr[4] = { o0, o1, o2, o3 };
        for (uint b = 0u; b < 4u; ++b) {
            uint jv = (b << 5u) + lane;
            float zv = z_h[jv];
            float out_v = (o_arr[b] * o_scale * ssm_norm[jv]) * (zv / (1.0f + exp(-zv)));
            uint col = h * 128u + jv;
            float rot = vxt_fwht32_lane(vxt_apply_col_sign(out_v, col), lane);
            x_rot_h_out[col] = half(rot * 0.0625f);
        }
    }
    if (tg_pos.x == 0u) {
        uint inner_dim = p.n_vh * 128u;
        for (uint col = inner_dim + tid; col < p.padded_inner; col += 256u) {
            x_rot_h_out[col] = half(0.0h);
        }
    }
    (void)x_rot_out;
}

[[max_total_threads_per_threadgroup(512)]]
kernel void vxt_dnet_m1_fused16(
    device const float* qkv_in          [[buffer(0)]],
    device const float* z_in            [[buffer(1)]],
    device const float* ssm_conv1d      [[buffer(2)]],
    device const float* ssm_a           [[buffer(3)]],
    device const float* ssm_dt_bias     [[buffer(4)]],
    device const float* ssm_norm        [[buffer(5)]],
    device float* conv_state            [[buffer(6)]],
    device half* ssm_state              [[buffer(7)]],
    device const half* ab_rot_f16       [[buffer(8)]],
    device const float* x_rot_in        [[buffer(9)]],
    device half* x_rot_h_out            [[buffer(10)]],
    constant TurboDeltaHeadsParams& p   [[buffer(11)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    if (h >= p.n_vh) return;

    uint qk_dim = p.n_vh * 128u;
    uint conv_ch = 3u * qk_dim;
    threadgroup float q_sh[128];
    threadgroup float k_sh[128];
    threadgroup float v_sh[128];
    threadgroup float o_sh[128];
    threadgroup float sg_ab[8];

    if (tid < 384u) {
        uint sub = tid >> 7u;
        uint d = tid & 127u;
        uint c = sub * qk_dim + h * 128u + d;
        device const float* w = ssm_conv1d + c * 4u;
        float h0 = conv_state[c];
        float h1 = conv_state[conv_ch + c];
        float h2 = conv_state[2u * conv_ch + c];
        float x_c = qkv_in[c];
        float acc = fma(x_c, w[3], fma(h2, w[2], fma(h1, w[1], h0 * w[0])));
        conv_state[c] = h1;
        conv_state[conv_ch + c] = h2;
        conv_state[2u * conv_ch + c] = x_c;
        float sv = acc / (1.0f + exp(-acc));
        if (sub == 0u) q_sh[d] = sv;
        else if (sub == 1u) k_sh[d] = sv;
        else v_sh[d] = sv;
    } else {
        uint ab_sg = sg_id - 12u;
        uint n4 = p.padded_embd >> 2u;
        uint ab_row = (ab_sg < 2u) ? h : (p.n_vh + h);
        uint ab_sub = ab_sg & 1u;
        device const half4* wr_ab4 = (device const half4*)(ab_rot_f16 + ulong(ab_row) * ulong(p.padded_embd));
        device const float4* xr4 = (device const float4*)x_rot_in;
        float acc_ab = 0.0f;
        for (uint c4 = ab_sub * 32u + lane; c4 < n4; c4 += 64u) {
            acc_ab += dot(float4(wr_ab4[c4]), xr4[c4]);
        }
        acc_ab = simd_sum(acc_ab);
        if (lane == 0u) sg_ab[ab_sg] = acc_ab;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float4 qv = ((threadgroup const float4*)q_sh)[lane];
        float sc = 0.08838834764831845f * rsqrt(simd_sum(dot(qv, qv)) + p.eps);
        ((threadgroup float4*)q_sh)[lane] = qv * sc;
    } else if (sg_id == 1u) {
        float4 kv = ((threadgroup const float4*)k_sh)[lane];
        float sc = rsqrt(simd_sum(dot(kv, kv)) + p.eps);
        ((threadgroup float4*)k_sh)[lane] = kv * sc;
    } else if (sg_id == 2u && lane == 0u) {
        float alpha_val = sg_ab[0] + sg_ab[1];
        float beta_val = sg_ab[2] + sg_ab[3];
        float beta = 1.0f / (1.0f + exp(-beta_val));
        float x_sp = alpha_val + ssm_dt_bias[h];
        float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? exp(x_sp) : log(1.0f + exp(x_sp)));
        sg_ab[4] = exp(sp * ssm_a[h]);
        sg_ab[5] = beta;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 q4 = ((threadgroup const float4*)q_sh)[lane];
    float4 k4 = ((threadgroup const float4*)k_sh)[lane];
    float decay = sg_ab[4];
    float beta = sg_ab[5];
    device half* sh_base = ssm_state + ulong(h) * 16384ul;
    uint base_jv = ((sg_id >> 2u) << 5u) + (sg_id & 3u);

    for (uint g = 0u; g < 2u; ++g) {
        uint jv0 = base_jv + (g << 4u);
        uint jv1 = jv0 + 4u;
        uint jv2 = jv0 + 8u;
        uint jv3 = jv0 + 12u;
        device half4* row0 = (device half4*)(sh_base + (jv0 << 7u));
        device half4* row1 = (device half4*)(sh_base + (jv1 << 7u));
        device half4* row2 = (device half4*)(sh_base + (jv2 << 7u));
        device half4* row3 = (device half4*)(sh_base + (jv3 << 7u));
        float4 r0 = float4(row0[lane]) * decay;
        float4 r1 = float4(row1[lane]) * decay;
        float4 r2 = float4(row2[lane]) * decay;
        float4 r3 = float4(row3[lane]) * decay;
        float4 sk4 = simd_sum(float4(dot(r0, k4), dot(r1, k4), dot(r2, k4), dot(r3, k4)));
        float4 d4 = (float4(v_sh[jv0], v_sh[jv1], v_sh[jv2], v_sh[jv3]) - sk4) * beta;
        r0 = fma(float4(d4.x), k4, r0);
        r1 = fma(float4(d4.y), k4, r1);
        r2 = fma(float4(d4.z), k4, r2);
        r3 = fma(float4(d4.w), k4, r3);
        row0[lane] = half4(r0);
        row1[lane] = half4(r1);
        row2[lane] = half4(r2);
        row3[lane] = half4(r3);
        float4 o4 = simd_sum(float4(dot(r0, q4), dot(r1, q4), dot(r2, q4), dot(r3, q4)));
        if (lane == 0u) {
            o_sh[jv0] = o4.x;
            o_sh[jv1] = o4.y;
            o_sh[jv2] = o4.z;
            o_sh[jv3] = o4.w;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float o0 = o_sh[lane], o1 = o_sh[32u + lane], o2 = o_sh[64u + lane], o3 = o_sh[96u + lane];
        float sq_o = simd_sum(fma(o3, o3, fma(o2, o2, fma(o1, o1, o0 * o0))));
        if (lane == 0u) sg_ab[6] = rsqrt(sq_o * 0.0078125f + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id < 4u) {
        float o_scale = sg_ab[6];
        uint jv = (sg_id << 5u) + lane;
        float zv = z_in[h * 128u + jv];
        float out_v = (o_sh[jv] * o_scale * ssm_norm[jv]) * (zv / (1.0f + exp(-zv)));
        uint col = h * 128u + jv;
        float rot = vxt_fwht32_lane(vxt_apply_col_sign(out_v, col), lane);
        x_rot_h_out[col] = half(rot * 0.0625f);
    }
    if (h == 0u) {
        for (uint col = qk_dim + tid; col < p.padded_inner; col += 512u) {
            x_rot_h_out[col] = half(0.0h);
        }
    }
}

// 8c. Single-Dispatch 64-TG 100%-Occupancy Fused Conv1D + DeltaNet + Last-TG Output Norm/FWHT32 (`M = 1`)
[[max_total_threads_per_threadgroup(128)]]
kernel void vxt_dnet_m1_fused64(
    device const float* qkv_in          [[buffer(0)]],
    device const float* z_in            [[buffer(1)]],
    device const float* ssm_conv1d      [[buffer(2)]],
    device const float* ssm_a           [[buffer(3)]],
    device const float* ssm_dt_bias     [[buffer(4)]],
    device const float* ssm_norm        [[buffer(5)]],
    device float* conv_state            [[buffer(6)]],
    device half* ssm_state              [[buffer(7)]],
    device const half* ab_rot_f16       [[buffer(8)]],
    device const float* x_rot_in        [[buffer(9)]],
    device half* x_rot_h_out            [[buffer(10)]],
    constant TurboDeltaHeadsParams& p   [[buffer(11)]],
    VXT_COHERENT_DEV float* o_all       [[buffer(12)]],
    device atomic_uint* sync_cnt        [[buffer(13)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint tg_id = tg_pos.x;
    uint h = tg_id >> 2u;
    uint sub = tg_id & 3u;
    if (h >= p.n_vh) return;

    uint qk_dim = p.n_vh * 128u;
    uint conv_ch = 3u * qk_dim;
    threadgroup float q_sh[128];
    threadgroup float k_sh[128];
    threadgroup float v_sub[32];
    threadgroup float sg_ab[6];
    threadgroup uint head_done;

    // 1. Read Conv1D history (without mutating conv_state yet!) + compute q, k, and this TG's 32 v channels
    for (uint sc = 0u; sc < 2u; ++sc) {
        uint c = sc * qk_dim + h * 128u + tid;
        device const float* w = ssm_conv1d + c * 4u;
        float acc = fma(qkv_in[c], w[3], fma(conv_state[2u * conv_ch + c], w[2], fma(conv_state[conv_ch + c], w[1], conv_state[c] * w[0])));
        float sv = acc / (1.0f + exp(-acc));
        if (sc == 0u) q_sh[tid] = sv;
        else k_sh[tid] = sv;
    }
    if (tid < 32u) {
        uint c = 2u * qk_dim + h * 128u + (sub << 5u) + tid;
        device const float* w = ssm_conv1d + c * 4u;
        float acc = fma(qkv_in[c], w[3], fma(conv_state[2u * conv_ch + c], w[2], fma(conv_state[conv_ch + c], w[1], conv_state[c] * w[0])));
        v_sub[tid] = acc / (1.0f + exp(-acc));
    }

    uint n4 = p.padded_embd >> 2u;
    uint ab_row = (sg_id < 2u) ? h : (p.n_vh + h);
    uint ab_sub = sg_id & 1u;
    device const half4* wr_ab4 = (device const half4*)(ab_rot_f16 + ulong(ab_row) * ulong(p.padded_embd));
    device const float4* xr4 = (device const float4*)x_rot_in;
    float acc_ab = 0.0f;
    for (uint c4 = ab_sub * 32u + lane; c4 < n4; c4 += 64u) {
        acc_ab += dot(float4(wr_ab4[c4]), xr4[c4]);
    }
    acc_ab = simd_sum(acc_ab);
    if (lane == 0u) sg_ab[sg_id] = acc_ab;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float4 qv = ((threadgroup const float4*)q_sh)[lane];
        float sc = 0.08838834764831845f * rsqrt(simd_sum(dot(qv, qv)) + p.eps);
        ((threadgroup float4*)q_sh)[lane] = qv * sc;
    } else if (sg_id == 1u) {
        float4 kv = ((threadgroup const float4*)k_sh)[lane];
        float sc = rsqrt(simd_sum(dot(kv, kv)) + p.eps);
        ((threadgroup float4*)k_sh)[lane] = kv * sc;
    } else if (sg_id == 2u && lane == 0u) {
        float alpha_val = sg_ab[0] + sg_ab[1];
        float beta_val = sg_ab[2] + sg_ab[3];
        float beta = 1.0f / (1.0f + exp(-beta_val));
        float x_sp = alpha_val + ssm_dt_bias[h];
        float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? exp(x_sp) : log(1.0f + exp(x_sp)));
        sg_ab[4] = exp(sp * ssm_a[h]);
        sg_ab[5] = beta;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Sweep 32 rows of FP16 ssm_state
    float4 q4 = ((threadgroup const float4*)q_sh)[lane];
    float4 k4 = ((threadgroup const float4*)k_sh)[lane];
    float decay = sg_ab[4];
    float beta = sg_ab[5];
    device half* sh_base = ssm_state + ulong(h) * 16384ul;
    VXT_COHERENT_DEV float* o_h = o_all + h * 128u;
    uint base_jv = (sub << 5u) + sg_id;

    for (uint g = 0u; g < 2u; ++g) {
        uint jv0 = base_jv + (g << 4u);
        uint jv1 = jv0 + 4u;
        uint jv2 = jv0 + 8u;
        uint jv3 = jv0 + 12u;
        device half4* row0 = (device half4*)(sh_base + (jv0 << 7u));
        device half4* row1 = (device half4*)(sh_base + (jv1 << 7u));
        device half4* row2 = (device half4*)(sh_base + (jv2 << 7u));
        device half4* row3 = (device half4*)(sh_base + (jv3 << 7u));
        float4 r0 = float4(row0[lane]) * decay;
        float4 r1 = float4(row1[lane]) * decay;
        float4 r2 = float4(row2[lane]) * decay;
        float4 r3 = float4(row3[lane]) * decay;
        float4 sk4 = simd_sum(float4(dot(r0, k4), dot(r1, k4), dot(r2, k4), dot(r3, k4)));
        float4 d4 = (float4(v_sub[jv0 & 31u], v_sub[jv1 & 31u], v_sub[jv2 & 31u], v_sub[jv3 & 31u]) - sk4) * beta;
        r0 = fma(float4(d4.x), k4, r0);
        r1 = fma(float4(d4.y), k4, r1);
        r2 = fma(float4(d4.z), k4, r2);
        r3 = fma(float4(d4.w), k4, r3);
        row0[lane] = half4(r0);
        row1[lane] = half4(r1);
        row2[lane] = half4(r2);
        row3[lane] = half4(r3);
        float4 o4 = simd_sum(float4(dot(r0, q4), dot(r1, q4), dot(r2, q4), dot(r3, q4)));
        if (lane == 0u) {
            o_h[jv0] = o4.x;
            o_h[jv1] = o4.y;
            o_h[jv2] = o4.z;
            o_h[jv3] = o4.w;
        }
    }

    // 3. Last-TG of head `h` (`prev == 3u`) updates conv_state & normalizes + FWHT32 -> x_rot_h_out
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0u) {
        vxt_device_fence();
        uint prev = atomic_fetch_add_explicit(&sync_cnt[p.sync_slot + h], 1u, memory_order_relaxed);
        if (prev == 3u) {
            atomic_store_explicit(&sync_cnt[p.sync_slot + h], 0u, memory_order_relaxed);
            head_done = 1u;
        } else {
            head_done = 0u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (head_done != 0u) {
        vxt_device_fence();
        for (uint sc = 0u; sc < 3u; ++sc) {
            uint c = sc * qk_dim + h * 128u + tid;
            float h1 = conv_state[conv_ch + c];
            float h2 = conv_state[2u * conv_ch + c];
            conv_state[c] = h1;
            conv_state[conv_ch + c] = h2;
            conv_state[2u * conv_ch + c] = qkv_in[c];
        }
        float ov = o_h[tid];
        float sq_o = simd_sum(ov * ov);
        if (lane == 0u) sg_ab[sg_id] = sq_o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u) {
            float s4 = (lane < 4u) ? sg_ab[lane] : 0.0f;
            s4 = simd_sum(s4);
            if (lane == 0u) sg_ab[0] = rsqrt(s4 * 0.0078125f + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float o_scale = sg_ab[0];
        float zv = z_in[h * 128u + tid];
        float out_v = (ov * o_scale * ssm_norm[tid]) * (zv / (1.0f + exp(-zv)));
        uint col = h * 128u + tid;
        float rot = vxt_fwht32_lane(vxt_apply_col_sign(out_v, col), lane);
        x_rot_h_out[col] = half(rot * 0.0625f);
        if (h == 0u) {
            for (uint pc = qk_dim + tid; pc < p.padded_inner; pc += 128u) {
                x_rot_h_out[pc] = half(0.0h);
            }
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_deltanet_fused_1to1_fp16(
    device const float* qkv_in          [[buffer(0)]],
    device const float* z_in            [[buffer(1)]],
    device float* alpha_beta_out        [[buffer(2)]],
    device const float* ssm_conv1d      [[buffer(3)]],
    device const float* ssm_a           [[buffer(4)]],
    device const float* ssm_dt_bias     [[buffer(5)]],
    device const float* ssm_norm        [[buffer(6)]],
    device float* conv_state            [[buffer(7)]],
    device half* ssm_state              [[buffer(8)]],
    device float* x_rot_out             [[buffer(9)]],
    device float* conv_bak              [[buffer(10)]],
    device half* ssm_bak                [[buffer(11)]],
    constant TurboDeltaHeadsParams& p   [[buffer(12)]],
    device const half* ab_rot_f16       [[buffer(13)]],
    device const float* x_rot_in        [[buffer(14)]],
    device half* x_rot_h_out            [[buffer(15)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    if (h >= p.n_vh) return;

    uint dk = p.dk;
    uint dv = p.dv;
    uint qk_dim = p.n_kh * dk;
    uint conv_ch = 2u * qk_dim + p.n_vh * dv;
    uint head_ch = 2u * dk + dv;

    threadgroup float q_sh[512];
    threadgroup float k_sh[512];
    threadgroup float v_sh[512];
    threadgroup float sg_red[16];

    device half* sh_base = ssm_state + ulong(h) * ulong(dv) * ulong(dk);
    device half* bk_base = ssm_bak + ulong(h) * ulong(dv) * ulong(dk);

    if (dk == 128u && dv == 128u && p.batch_m <= 4u) {
        uint n4 = p.padded_embd >> 2u;
        device const half4* wr_ab4 = (device const half4*)(
            ab_rot_f16 + ulong((sg_id < 4u) ? h : (p.n_vh + h)) * ulong(p.padded_embd)
        );
        uint ab_sg = sg_id & 3u;

        for (uint m = 0u; m < p.batch_m; ++m) {
            for (uint idx = tid; idx < 384u; idx += 256u) {
                uint c = (idx < 128u)
                    ? (h * 128u + idx)
                    : ((idx < 256u) ? (qk_dim + h * 128u + (idx - 128u)) : (2u * qk_dim + h * 128u + (idx - 256u)));
                device const float* w = ssm_conv1d + c * 4u;
                float h0 = conv_state[c];
                float h1 = conv_state[conv_ch + c];
                float h2 = conv_state[2u * conv_ch + c];
                float x_c = qkv_in[ulong(m) * ulong(p.qkv_stride) + ulong(c)];
                float acc = fma(x_c, w[3], fma(h2, w[2], fma(h1, w[1], h0 * w[0])));
                conv_state[c] = h1;
                conv_state[conv_ch + c] = h2;
                conv_state[2u * conv_ch + c] = x_c;
                if (p.save_backup != 0u && m == 0u) {
                    conv_bak[c] = h1;
                    conv_bak[conv_ch + c] = h2;
                    conv_bak[2u * conv_ch + c] = x_c;
                }
                float sv = acc / (1.0f + exp(-acc));
                uint m_off = m << 7u;
                if (idx < 128u) q_sh[m_off + idx] = sv;
                else if (idx < 256u) k_sh[m_off + (idx - 128u)] = sv;
                else v_sh[m_off + (idx - 256u)] = sv;
            }

            device const float4* xr4 = (device const float4*)(x_rot_in + ulong(m) * ulong(p.padded_embd));
            float acc_ab = 0.0f;
            for (uint c4 = ab_sg * 32u + lane; c4 < n4; c4 += 128u) {
                acc_ab += dot(float4(wr_ab4[c4]), xr4[c4]);
            }
            acc_ab = simd_sum(acc_ab);
            if (lane == 0u) sg_red[8u + sg_id] = acc_ab;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint m_off = m << 7u;
            if (sg_id == 0u) {
                float4 qv = *(threadgroup const float4*)(q_sh + m_off + (lane << 2u));
                float sq = simd_sum(dot(qv, qv));
                float sc = 0.08838834764831845f * rsqrt(sq + p.eps);
                *(threadgroup float4*)(q_sh + m_off + (lane << 2u)) = qv * sc;
            } else if (sg_id == 1u) {
                float4 kv = *(threadgroup const float4*)(k_sh + m_off + (lane << 2u));
                float sq = simd_sum(dot(kv, kv));
                float sc = rsqrt(sq + p.eps);
                *(threadgroup float4*)(k_sh + m_off + (lane << 2u)) = kv * sc;
            } else if (sg_id == 2u && lane == 0u) {
                float alpha_val = (sg_red[8] + sg_red[9]) + (sg_red[10] + sg_red[11]);
                float beta_val = (sg_red[12] + sg_red[13]) + (sg_red[14] + sg_red[15]);
                if (p.save_backup != 0u) {
                    alpha_beta_out[ulong(m) * ulong(2u * p.n_vh) + ulong(h)] = alpha_val;
                    alpha_beta_out[ulong(m) * ulong(2u * p.n_vh) + ulong(p.n_vh + h)] = beta_val;
                }
                float beta = 1.0f / (1.0f + exp(-beta_val));
                float x_sp = alpha_val + ssm_dt_bias[h];
                float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? exp(x_sp) : log(1.0f + exp(x_sp)));
                sg_red[m * 2u] = exp(sp * ssm_a[h]);
                sg_red[m * 2u + 1u] = beta;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint g = 0u; g < 4u; ++g) {
            uint jv0 = sg_id + (g << 5u);
            uint jv1 = jv0 + 8u;
            uint jv2 = jv0 + 16u;
            uint jv3 = jv0 + 24u;
            device half4* row0 = (device half4*)(sh_base + (jv0 << 7u));
            device half4* row1 = (device half4*)(sh_base + (jv1 << 7u));
            device half4* row2 = (device half4*)(sh_base + (jv2 << 7u));
            device half4* row3 = (device half4*)(sh_base + (jv3 << 7u));
            float4 r0 = float4(row0[lane]);
            float4 r1 = float4(row1[lane]);
            float4 r2 = float4(row2[lane]);
            float4 r3 = float4(row3[lane]);
            half4 h0, h1, h2, h3;

            for (uint m = 0u; m < p.batch_m; ++m) {
                float decay = sg_red[m * 2u];
                float beta = sg_red[m * 2u + 1u];
                uint m_off = m << 7u;
                uint b4 = m_off + (lane << 2u);
                float4 k4 = *(threadgroup const float4*)(k_sh + b4);
                float4 q4 = *(threadgroup const float4*)(q_sh + b4);
                r0 *= decay; r1 *= decay; r2 *= decay; r3 *= decay;
                float4 sk4 = simd_sum(float4(dot(r0, k4), dot(r1, k4), dot(r2, k4), dot(r3, k4)));
                float4 d4 = (float4(v_sh[m_off + jv0], v_sh[m_off + jv1], v_sh[m_off + jv2], v_sh[m_off + jv3]) - sk4) * beta;
                r0 = fma(float4(d4.x), k4, r0);
                r1 = fma(float4(d4.y), k4, r1);
                r2 = fma(float4(d4.z), k4, r2);
                r3 = fma(float4(d4.w), k4, r3);
                float4 o4 = simd_sum(float4(dot(r0, q4), dot(r1, q4), dot(r2, q4), dot(r3, q4)));
                if (lane == 0u) {
                    v_sh[m_off + jv0] = o4.x;
                    v_sh[m_off + jv1] = o4.y;
                    v_sh[m_off + jv2] = o4.z;
                    v_sh[m_off + jv3] = o4.w;
                }
                h0 = half4(r0); h1 = half4(r1); h2 = half4(r2); h3 = half4(r3);
                if (p.save_backup != 0u && m == 0u) {
                    ((device half4*)(bk_base + (jv0 << 7u)))[lane] = h0;
                    ((device half4*)(bk_base + (jv1 << 7u)))[lane] = h1;
                    ((device half4*)(bk_base + (jv2 << 7u)))[lane] = h2;
                    ((device half4*)(bk_base + (jv3 << 7u)))[lane] = h3;
                }
                if (m + 1u < p.batch_m) {
                    r0 = float4(h0); r1 = float4(h1); r2 = float4(h2); r3 = float4(h3);
                }
            }
            row0[lane] = h0;
            row1[lane] = h1;
            row2[lane] = h2;
            row3[lane] = h3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id < p.batch_m) {
            uint m = sg_id;
            uint m_off = m << 7u;
            float o0 = v_sh[m_off + lane];
            float o1 = v_sh[m_off + 32u + lane];
            float o2 = v_sh[m_off + 64u + lane];
            float o3 = v_sh[m_off + 96u + lane];
            float sq_o = simd_sum(fma(o3, o3, fma(o2, o2, fma(o1, o1, o0 * o0))));
            float o_scale = rsqrt(sq_o * 0.0078125f + p.eps);
            device const float* z_h = z_in + ulong(m) * ulong(p.z_stride) + ulong(h * 128u);
            device float* rot_m = x_rot_out + ulong(m) * ulong(p.padded_inner);
            device half* rot_hm = x_rot_h_out + ulong(m) * ulong(p.padded_inner);
            float o_arr[4] = { o0, o1, o2, o3 };
            for (uint b = 0u; b < 4u; ++b) {
                uint jv = (b << 5u) + lane;
                float zv = z_h[jv];
                float out_v = (o_arr[b] * o_scale * ssm_norm[jv]) * (zv / (1.0f + exp(-zv)));
                uint col = h * 128u + jv;
                float rot = vxt_fwht32_lane(vxt_apply_col_sign(out_v, col), lane);
                rot_m[col] = rot;
                rot_hm[col] = half(rot * 0.0625f);
            }
        }
        if (h + 1u == p.n_vh) {
            uint inner_dim = p.n_vh * 128u;
            for (uint m = 0u; m < p.batch_m; ++m) {
                device float* rot_m = x_rot_out + ulong(m) * ulong(p.padded_inner);
                device half* rot_hm = x_rot_h_out + ulong(m) * ulong(p.padded_inner);
                for (uint col = inner_dim + tid; col < p.padded_inner; col += 256u) {
                    rot_m[col] = 0.0f;
                    rot_hm[col] = half(0.0h);
                }
            }
        }
        return;
    }

    threadgroup float* o_sh = v_sh;
    for (uint m = 0u; m < p.batch_m; ++m) {
        for (uint idx = tid; idx < head_ch; idx += 256u) {
            uint c = (idx < dk)
                ? (h * dk + idx)
                : ((idx < 2u * dk) ? (qk_dim + h * dk + (idx - dk)) : (2u * qk_dim + h * dv + (idx - 2u * dk)));
            device const float* w = ssm_conv1d + c * 4u;
            float h0 = conv_state[c];
            float h1 = conv_state[conv_ch + c];
            float h2 = conv_state[2u * conv_ch + c];
            float x_c = qkv_in[ulong(m) * ulong(p.qkv_stride) + ulong(c)];
            float acc = fma(x_c, w[3], fma(h2, w[2], fma(h1, w[1], h0 * w[0])));
            conv_state[c] = h1;
            conv_state[conv_ch + c] = h2;
            conv_state[2u * conv_ch + c] = x_c;
            if (p.save_backup != 0u && m == 0u) {
                conv_bak[c] = h1;
                conv_bak[conv_ch + c] = h2;
                conv_bak[2u * conv_ch + c] = x_c;
            }
            float sv = acc / (1.0f + exp(-acc));
            if (idx < dk) q_sh[idx] = sv;
            else if (idx < 2u * dk) k_sh[idx - dk] = sv;
            else v_sh[idx - 2u * dk] = sv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {
            float sq = 0.0f;
            for (uint i = lane; i < dk; i += 32u) {
                float v = q_sh[i];
                sq = fma(v, v, sq);
            }
            sq = simd_sum(sq);
            float sc = rsqrt(float(dk)) * rsqrt(sq + p.eps);
            for (uint i = lane; i < dk; i += 32u) q_sh[i] *= sc;
        } else if (sg_id == 1u) {
            float sq = 0.0f;
            for (uint i = lane; i < dk; i += 32u) {
                float v = k_sh[i];
                sq = fma(v, v, sq);
            }
            sq = simd_sum(sq);
            float sc = rsqrt(sq + p.eps);
            for (uint i = lane; i < dk; i += 32u) k_sh[i] *= sc;
        } else if (sg_id == 2u) {
            device const half4* wr4 = (device const half4*)(ab_rot_f16 + ulong(h) * ulong(p.padded_embd));
            device const float4* xr4 = (device const float4*)(x_rot_in + ulong(m) * ulong(p.padded_embd));
            uint n4 = p.padded_embd >> 2u;
            float acc_a = 0.0f;
            for (uint c4 = lane; c4 < n4; c4 += 32u) {
                acc_a += dot(float4(wr4[c4]), xr4[c4]);
            }
            acc_a = simd_sum(acc_a);
            if (lane == 0u) {
                sg_red[8] = acc_a;
                if (p.save_backup != 0u) alpha_beta_out[ulong(m) * ulong(2u * p.n_vh) + ulong(h)] = acc_a;
            }
        } else if (sg_id == 3u) {
            device const half4* wr4 = (device const half4*)(ab_rot_f16 + ulong(p.n_vh + h) * ulong(p.padded_embd));
            device const float4* xr4 = (device const float4*)(x_rot_in + ulong(m) * ulong(p.padded_embd));
            uint n4 = p.padded_embd >> 2u;
            float acc_b = 0.0f;
            for (uint c4 = lane; c4 < n4; c4 += 32u) {
                acc_b += dot(float4(wr4[c4]), xr4[c4]);
            }
            acc_b = simd_sum(acc_b);
            if (lane == 0u) {
                sg_red[9] = acc_b;
                if (p.save_backup != 0u) alpha_beta_out[ulong(m) * ulong(2u * p.n_vh) + ulong(p.n_vh + h)] = acc_b;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float alpha_val = sg_red[8];
        float beta_val = sg_red[9];
        float beta = 1.0f / (1.0f + exp(-beta_val));
        float x_sp = alpha_val + ssm_dt_bias[h];
        float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? exp(x_sp) : log(1.0f + exp(x_sp)));
        float decay = exp(sp * ssm_a[h]);

        if (dk == 128u) {
            uint b4 = lane << 2u;
            float4 k4 = *(threadgroup const float4*)(k_sh + b4);
            float4 q4 = *(threadgroup const float4*)(q_sh + b4);
            for (uint jv = sg_id; jv < dv; jv += 8u) {
                device half4* row4 = (device half4*)(sh_base + (jv << 7u));
                float4 r4 = float4(row4[lane]) * decay;
                float sk = simd_sum(dot(r4, k4));
                float d_jv = (v_sh[jv] - sk) * beta;
                r4 = fma(float4(d_jv), k4, r4);
                half4 rh = half4(r4);
                row4[lane] = rh;
                if (p.save_backup != 0u && m == 0u) {
                    ((device half4*)(bk_base + (jv << 7u)))[lane] = rh;
                }
                float o_jv = simd_sum(dot(r4, q4));
                if (lane == 0u) o_sh[jv] = o_jv;
            }
        } else {
            uint dk_chunks = (dk + 31u) >> 5u;
            for (uint jv = sg_id; jv < dv; jv += 8u) {
                device half* row = sh_base + ulong(jv) * ulong(dk);
                device half* brow = bk_base + ulong(jv) * ulong(dk);
                float r_reg[16];
                float sk = 0.0f;
                for (uint c = 0u; c < dk_chunks; ++c) {
                    uint ik = (c << 5u) + lane;
                    if (ik < dk) {
                        float sd = float(row[ik]) * decay;
                        r_reg[c] = sd;
                        sk = fma(sd, k_sh[ik], sk);
                    }
                }
                sk = simd_sum(sk);
                float d_jv = (v_sh[jv] - sk) * beta;
                float o_jv = 0.0f;
                for (uint c = 0u; c < dk_chunks; ++c) {
                    uint ik = (c << 5u) + lane;
                    if (ik < dk) {
                        float sn = fma(d_jv, k_sh[ik], r_reg[c]);
                        half shv = half(sn);
                        row[ik] = shv;
                        if (p.save_backup != 0u && m == 0u) brow[ik] = shv;
                        o_jv = fma(sn, q_sh[ik], o_jv);
                    }
                }
                o_jv = simd_sum(o_jv);
                if (lane == 0u) o_sh[jv] = o_jv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sq_o = 0.0f;
        for (uint jv = tid; jv < dv; jv += 256u) {
            float v = o_sh[jv];
            sq_o = fma(v, v, sq_o);
        }
        sq_o = simd_sum(sq_o);
        if (lane == 0u) sg_red[sg_id] = sq_o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u) {
            float v = (lane < 8u) ? sg_red[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0u) sg_red[0] = rsqrt(v / float(dv) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float o_scale = sg_red[0];
        device const float* z_h = z_in + ulong(m) * ulong(p.z_stride) + ulong(h * dv);
        device float* rot_m = x_rot_out + ulong(m) * ulong(p.padded_inner);
        device half* rot_hm = x_rot_h_out + ulong(m) * ulong(p.padded_inner);

        uint dv_blks = dv >> 5u;
        for (uint b = sg_id; b < dv_blks; b += 8u) {
            uint jv = (b << 5u) + lane;
            float zv = z_h[jv];
            float out_v = (o_sh[jv] * o_scale * ssm_norm[jv]) * (zv / (1.0f + exp(-zv)));
            uint col = h * dv + jv;
            float rot = vxt_fwht32_lane(vxt_apply_col_sign(out_v, col), lane);
            rot_m[col] = rot;
            rot_hm[col] = half(rot * 0.0625f);
        }
        if (h + 1u == p.n_vh) {
            uint inner_dim = p.n_vh * dv;
            for (uint col = inner_dim + tid; col < p.padded_inner; col += 256u) {
                rot_m[col] = 0.0f;
                rot_hm[col] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void vxt_deltanet_heads_fp16_turbo(
    device const float* conv_silu       [[buffer(0)]],
    device const float* qkv_gate        [[buffer(1)]],
    device const float* alpha_beta      [[buffer(2)]],
    device const float* ssm_a           [[buffer(3)]],
    device const float* ssm_dt_bias     [[buffer(4)]],
    device const float* ssm_norm        [[buffer(5)]],
    device half* ssm_state              [[buffer(6)]],
    device half* ssm_bak                [[buffer(7)]],
    device float* x_rot_out             [[buffer(8)]],
    constant TurboDeltaHeadsParams& p   [[buffer(9)]],
    device half* x_rot_h_out            [[buffer(10)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    if (h >= p.n_vh) return;

    uint dk = p.dk;
    uint dv = p.dv;
    uint hk = h / (p.n_vh / p.n_kh);
    uint qk_dim = p.n_kh * dk;
    uint conv_ch = 2u * qk_dim + p.n_vh * dv;

    threadgroup float q_sh[512];
    threadgroup float k_sh[512];
    threadgroup float o_sh[512];
    threadgroup float sg_red[8];

    device half* sh_base = ssm_state + ulong(h) * ulong(dv) * ulong(dk);
    device half* bk_base = ssm_bak + ulong(h) * ulong(dv) * ulong(dk);

    for (uint m = 0u; m < p.batch_m; ++m) {
        device const float* cs_m = conv_silu + ulong(m) * ulong(conv_ch);
        device const float* q_src = cs_m + hk * dk;
        device const float* k_src = cs_m + qk_dim + hk * dk;
        device const float* v_h = cs_m + 2u * qk_dim + h * dv;

        if (sg_id == 0u) {
            float sq = 0.0f;
            for (uint i = lane; i < dk; i += 32u) {
                float v = q_src[i];
                sq = fma(v, v, sq);
            }
            sq = simd_sum(sq);
            float sc = rsqrt(float(dk)) * rsqrt(sq + p.eps);
            for (uint i = lane; i < dk; i += 32u) q_sh[i] = q_src[i] * sc;
        } else if (sg_id == 1u) {
            float sq = 0.0f;
            for (uint i = lane; i < dk; i += 32u) {
                float v = k_src[i];
                sq = fma(v, v, sq);
            }
            sq = simd_sum(sq);
            float sc = rsqrt(sq + p.eps);
            for (uint i = lane; i < dk; i += 32u) k_sh[i] = k_src[i] * sc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float* ab_m = alpha_beta + ulong(m) * ulong(2u * p.n_vh);
        float alpha_val = ab_m[h];
        float beta_val = ab_m[p.n_vh + h];
        float beta = 1.0f / (1.0f + exp(-beta_val));
        float x_sp = alpha_val + ssm_dt_bias[h];
        float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? exp(x_sp) : log(1.0f + exp(x_sp)));
        float decay = exp(sp * ssm_a[h]);

        if (dk == 128u) {
            uint i0 = lane, i1 = lane + 32u, i2 = lane + 64u, i3 = lane + 96u;
            float k0 = k_sh[i0], k1 = k_sh[i1], k2 = k_sh[i2], k3 = k_sh[i3];
            float q0 = q_sh[i0], q1 = q_sh[i1], q2 = q_sh[i2], q3 = q_sh[i3];
            for (uint jv = sg_id; jv < dv; jv += 8u) {
                device half* row = sh_base + (jv << 7u);
                float r0 = float(row[i0]) * decay;
                float r1 = float(row[i1]) * decay;
                float r2 = float(row[i2]) * decay;
                float r3 = float(row[i3]) * decay;
                float sk = simd_sum(fma(r3, k3, fma(r2, k2, fma(r1, k1, r0 * k0))));
                float d_jv = (v_h[jv] - sk) * beta;
                r0 = fma(d_jv, k0, r0);
                r1 = fma(d_jv, k1, r1);
                r2 = fma(d_jv, k2, r2);
                r3 = fma(d_jv, k3, r3);
                row[i0] = half(r0);
                row[i1] = half(r1);
                row[i2] = half(r2);
                row[i3] = half(r3);
                if (p.save_backup != 0u && m == 0u) {
                    device half* brow = bk_base + (jv << 7u);
                    brow[i0] = half(r0);
                    brow[i1] = half(r1);
                    brow[i2] = half(r2);
                    brow[i3] = half(r3);
                }
                float o_jv = simd_sum(fma(r3, q3, fma(r2, q2, fma(r1, q1, r0 * q0))));
                if (lane == 0u) o_sh[jv] = o_jv;
            }
        } else {
            uint dk_chunks = (dk + 31u) >> 5u;
            for (uint jv = sg_id; jv < dv; jv += 8u) {
                device half* row = sh_base + ulong(jv) * ulong(dk);
                device half* brow = bk_base + ulong(jv) * ulong(dk);
                float r_reg[16];
                float sk = 0.0f;
                for (uint c = 0u; c < dk_chunks; ++c) {
                    uint ik = (c << 5u) + lane;
                    if (ik < dk) {
                        float sd = float(row[ik]) * decay;
                        r_reg[c] = sd;
                        sk = fma(sd, k_sh[ik], sk);
                    }
                }
                sk = simd_sum(sk);
                float d_jv = (v_h[jv] - sk) * beta;
                float o_jv = 0.0f;
                for (uint c = 0u; c < dk_chunks; ++c) {
                    uint ik = (c << 5u) + lane;
                    if (ik < dk) {
                        float sn = fma(d_jv, k_sh[ik], r_reg[c]);
                        half hsn = half(sn);
                        row[ik] = hsn;
                        if (p.save_backup != 0u && m == 0u) brow[ik] = hsn;
                        o_jv = fma(sn, q_sh[ik], o_jv);
                    }
                }
                o_jv = simd_sum(o_jv);
                if (lane == 0u) o_sh[jv] = o_jv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sq_o = 0.0f;
        for (uint jv = tid; jv < dv; jv += 256u) {
            float v = o_sh[jv];
            sq_o = fma(v, v, sq_o);
        }
        sq_o = simd_sum(sq_o);
        if (lane == 0u) sg_red[sg_id] = sq_o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u) {
            float v = (lane < 8u) ? sg_red[lane] : 0.0f;
            v = simd_sum(v);
            if (lane == 0u) sg_red[0] = rsqrt(v / float(dv) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float o_scale = sg_red[0];
        device const float* z_h = qkv_gate + ulong(m) * ulong(p.z_stride) + ulong(h * dv);
        device float* rot_m = x_rot_out + ulong(m) * ulong(p.padded_inner);
        device half* rot_hm = x_rot_h_out + ulong(m) * ulong(p.padded_inner);

        uint dv_blks = dv >> 5u;
        for (uint b = sg_id; b < dv_blks; b += 8u) {
            uint jv = (b << 5u) + lane;
            float zv = z_h[jv];
            float out_v = (o_sh[jv] * o_scale * ssm_norm[jv]) * (zv / (1.0f + exp(-zv)));
            uint col = h * dv + jv;
            float rot = vxt_fwht32_lane(vxt_apply_col_sign(out_v, col), lane);
            rot_m[col] = rot;
            rot_hm[col] = half(rot * 0.0625f);
        }
        if (h + 1u == p.n_vh) {
            uint inner_dim = p.n_vh * dv;
            for (uint col = inner_dim + tid; col < p.padded_inner; col += 256u) {
                rot_m[col] = 0.0f;
                rot_hm[col] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// 9. Batched FP16-KV FullGqa Prep + Flash-Decoding + Fused FWHT32 -> x_rot_out
kernel void vxt_gqa_prep_fp16_batch(
    device const float* qkv_gqa [[buffer(0)]],
    device const float* q_norm  [[buffer(1)]],
    device const float* k_norm  [[buffer(2)]],
    device const float* q_bias  [[buffer(3)]],
    device const float* k_bias  [[buffer(4)]],
    device const float* v_bias  [[buffer(5)]],
    device float* q_cur_out     [[buffer(6)]],
    device float* q_gate_out    [[buffer(7)]],
    device half* k_cache        [[buffer(8)]],
    device half* v_cache        [[buffer(9)]],
    device const float* kv_gqa  [[buffer(10)]],
    constant TurboGqaParams& p  [[buffer(11)]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint3 tg_pos                [[threadgroup_position_in_grid]]
) {
    uint idx = tg_pos.x;
    uint m = tg_pos.y;
    if (m >= p.batch_m) return;

    uint hd = p.hd;
    uint pos = p.pos_base + m;
    bool gated_q = (p.flags & 1u) != 0u;
    bool neox = (p.flags & 2u) != 0u;
    bool has_qn = (p.flags & 4u) != 0u;
    bool has_kn = (p.flags & 8u) != 0u;
    bool has_qb = (p.flags & 16u) != 0u;
    bool has_kb = (p.flags & 32u) != 0u;
    bool has_vb = (p.flags & 64u) != 0u;

    device const float* q_raw = qkv_gqa + ulong(m) * ulong(p.qkv_stride);
    device const float* k_cur = kv_gqa + ulong(m) * ulong(p.kv_stride);
    device const float* v_cur = k_cur + p.nkv * hd;
    uint q_dim = p.nh * hd;

    threadgroup float h_buf[512];

    if (idx < p.nh) {
        uint h = idx;
        if (gated_q) {
            uint base = h * 2u * hd;
            for (uint d = lane; d < hd; d += 32u) {
                float qb0 = has_qb ? q_bias[base + d] : 0.0f;
                float qb1 = has_qb ? q_bias[base + hd + d] : 0.0f;
                h_buf[d] = q_raw[base + d] + qb0;
                q_gate_out[ulong(m) * ulong(q_dim) + ulong(h * hd + d)] = q_raw[base + hd + d] + qb1;
            }
        } else {
            uint base = h * hd;
            for (uint d = lane; d < hd; d += 32u) {
                h_buf[d] = q_raw[base + d] + (has_qb ? q_bias[base + d] : 0.0f);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (has_qn) {
            float sq = 0.0f;
            for (uint d = lane; d < hd; d += 32u) {
                float v = h_buf[d];
                sq = fma(v, v, sq);
            }
            sq = simd_sum(sq);
            float sc = rsqrt(sq / float(hd) + p.eps);
            for (uint d = lane; d < hd; d += 32u) h_buf[d] *= sc * q_norm[d];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint rdim = min(p.rope_dim, hd);
        uint half_r = rdim >> 1u;
        float inv_rdim_log2 = (rdim > 0u) ? (-2.0f * log2(p.rope_freq_base) / float(rdim)) : 0.0f;
        for (uint i = lane; i < half_r; i += 32u) {
            float theta = float(pos) * exp2(float(i) * inv_rdim_log2);
            float cs = cos(theta);
            float sn = sin(theta);
            uint i0 = neox ? i : (2u * i);
            uint i1 = neox ? (i + half_r) : (2u * i + 1u);
            float x0 = h_buf[i0];
            float x1 = h_buf[i1];
            h_buf[i0] = x0 * cs - x1 * sn;
            h_buf[i1] = x0 * sn + x1 * cs;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lane; d < hd; d += 32u) {
            q_cur_out[ulong(m) * ulong(q_dim) + ulong(h * hd + d)] = h_buf[d];
        }
    } else if (idx < p.nh + p.nkv) {
        uint kh = idx - p.nh;
        uint base = kh * hd;
        ulong cache_off = ulong(pos) * ulong(p.nkv * hd) + ulong(base);
        for (uint d = lane; d < hd; d += 32u) {
            h_buf[d] = k_cur[base + d] + (has_kb ? k_bias[base + d] : 0.0f);
            v_cache[cache_off + d] = half(v_cur[base + d] + (has_vb ? v_bias[base + d] : 0.0f));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (has_kn) {
            float sq = 0.0f;
            for (uint d = lane; d < hd; d += 32u) {
                float v = h_buf[d];
                sq = fma(v, v, sq);
            }
            sq = simd_sum(sq);
            float sc = rsqrt(sq / float(hd) + p.eps);
            for (uint d = lane; d < hd; d += 32u) h_buf[d] *= sc * k_norm[d];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint rdim = min(p.rope_dim, hd);
        uint half_r = rdim >> 1u;
        float inv_rdim_log2 = (rdim > 0u) ? (-2.0f * log2(p.rope_freq_base) / float(rdim)) : 0.0f;
        for (uint i = lane; i < half_r; i += 32u) {
            float theta = float(pos) * exp2(float(i) * inv_rdim_log2);
            float cs = cos(theta);
            float sn = sin(theta);
            uint i0 = neox ? i : (2u * i);
            uint i1 = neox ? (i + half_r) : (2u * i + 1u);
            float x0 = h_buf[i0];
            float x1 = h_buf[i1];
            h_buf[i0] = x0 * cs - x1 * sn;
            h_buf[i1] = x0 * sn + x1 * cs;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint d = lane; d < hd; d += 32u) {
            k_cache[cache_off + d] = half(h_buf[d]);
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_gqa_attn_fp16_turbo(
    device const float* q_cur   [[buffer(0)]],
    device const float* q_gate  [[buffer(1)]],
    device const half* k_cache  [[buffer(2)]],
    device const half* v_cache  [[buffer(3)]],
    device float* x_rot_out     [[buffer(4)]],
    constant TurboGqaParams& p  [[buffer(5)]],
    device half* x_rot_h_out    [[buffer(6)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint3 tg_pos                [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    uint m = tg_pos.y;
    if (h >= p.nh || m >= p.batch_m) return;

    uint hd = p.hd;
    uint q_dim = p.nh * hd;
    uint kv_h = h / (p.nh / p.nkv);
    uint seq_len = p.pos_base + m + 1u;
    ulong kv_stride = ulong(p.nkv * hd);
    float scale = rsqrt(float(hd));
    bool gated_q = (p.flags & 1u) != 0u;

    threadgroup float q_sh[512];
    threadgroup float sg_m[8];
    threadgroup float sg_l[8];
    threadgroup float sg_acc[8 * 512];

    device const float* q_m = q_cur + ulong(m) * ulong(q_dim);
    device const float* qg_m = q_gate + ulong(m) * ulong(q_dim);

    for (uint d = tid; d < hd; d += 256u) {
        q_sh[d] = q_m[h * hd + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i = -INFINITY;
    float l_i = 0.0f;

    if (hd == 256u) {
        uint d4 = lane << 2u;
        float4 qf0 = *(threadgroup const float4*)(q_sh + d4);
        float4 qf1 = *(threadgroup const float4*)(q_sh + 128u + d4);
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        for (uint t = sg_id; t < seq_len; t += 8u) {
            ulong base = ulong(t) * kv_stride + ulong(kv_h * 256u);
            device const half4* kh4 = (device const half4*)(k_cache + base);
            device const half4* vh4 = (device const half4*)(v_cache + base);
            half4 k0 = kh4[lane], k1 = kh4[lane + 32u];
            float dot_v = dot(qf0, float4(k0)) + dot(qf1, float4(k1));
            float s = simd_sum(dot_v) * scale;
            float m_new = max(m_i, s);
            float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
            float w = exp(s - m_new);
            l_i = fma(l_i, alpha, w);
            m_i = m_new;
            half4 v0 = vh4[lane], v1 = vh4[lane + 32u];
            acc0 = fma(float4(w), float4(v0), acc0 * alpha);
            acc1 = fma(float4(w), float4(v1), acc1 * alpha);
        }
        *(threadgroup float4*)(sg_acc + sg_id * 512u + d4) = acc0;
        *(threadgroup float4*)(sg_acc + sg_id * 512u + 128u + d4) = acc1;
    } else if (hd == 128u) {
        uint d4 = lane << 2u;
        float4 qf0 = *(threadgroup const float4*)(q_sh + d4);
        float4 acc0 = float4(0.0f);
        for (uint t = sg_id; t < seq_len; t += 8u) {
            ulong base = ulong(t) * kv_stride + ulong(kv_h * 128u);
            device const half4* kh4 = (device const half4*)(k_cache + base);
            device const half4* vh4 = (device const half4*)(v_cache + base);
            float dot_v = dot(qf0, float4(kh4[lane]));
            float s = simd_sum(dot_v) * scale;
            float m_new = max(m_i, s);
            float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
            float w = exp(s - m_new);
            l_i = fma(l_i, alpha, w);
            m_i = m_new;
            acc0 = fma(float4(w), float4(vh4[lane]), acc0 * alpha);
        }
        *(threadgroup float4*)(sg_acc + sg_id * 512u + d4) = acc0;
    } else {
        uint hd_chunks = (hd + 31u) >> 5u;
        float acc[16];
        for (uint c = 0u; c < hd_chunks; ++c) acc[c] = 0.0f;
        for (uint t = sg_id; t < seq_len; t += 8u) {
            ulong base = ulong(t) * kv_stride + ulong(kv_h * hd);
            device const half* kh = k_cache + base;
            device const half* vh = v_cache + base;
            float dot_v = 0.0f;
            for (uint c = 0u; c < hd_chunks; ++c) {
                uint d = (c << 5u) + lane;
                if (d < hd) dot_v = fma(q_sh[d], float(kh[d]), dot_v);
            }
            float s = simd_sum(dot_v) * scale;
            float m_new = max(m_i, s);
            float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
            float w = exp(s - m_new);
            l_i = fma(l_i, alpha, w);
            m_i = m_new;
            for (uint c = 0u; c < hd_chunks; ++c) {
                uint d = (c << 5u) + lane;
                if (d < hd) acc[c] = fma(w, float(vh[d]), acc[c] * alpha);
            }
        }
        for (uint c = 0u; c < hd_chunks; ++c) {
            uint d = (c << 5u) + lane;
            if (d < hd) sg_acc[sg_id * 512u + d] = acc[c];
        }
    }

    if (lane == 0u) {
        sg_m[sg_id] = m_i;
        sg_l[sg_id] = l_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_all = -INFINITY;
    for (uint sg = 0u; sg < 8u; ++sg) {
        if (sg_l[sg] > 0.0f) m_all = max(m_all, sg_m[sg]);
    }
    float w_sg[8];
    float l_all = 0.0f;
    for (uint sg = 0u; sg < 8u; ++sg) {
        float w = (sg_l[sg] > 0.0f) ? exp(sg_m[sg] - m_all) : 0.0f;
        w_sg[sg] = w;
        l_all = fma(sg_l[sg], w, l_all);
    }
    float inv_l = 1.0f / l_all;

    device float* rot_m = x_rot_out + ulong(m) * ulong(p.padded_qdim);
    device half* rot_hm = x_rot_h_out + ulong(m) * ulong(p.padded_qdim);
    uint hd_blks = hd >> 5u;
    for (uint b = sg_id; b < hd_blks; b += 8u) {
        uint d = (b << 5u) + lane;
        float out_d = 0.0f;
        for (uint sg = 0u; sg < 8u; ++sg) {
            out_d = fma(sg_acc[sg * 512u + d], w_sg[sg], out_d);
        }
        out_d *= inv_l;
        if (gated_q) {
            float g = qg_m[h * hd + d];
            out_d *= 1.0f / (1.0f + exp(-g));
        }
        uint col = h * hd + d;
        float rot = vxt_fwht32_lane(vxt_apply_col_sign(out_d, col), lane);
        rot_m[col] = rot;
        rot_hm[col] = half(rot * 0.0625f);
    }
    if (h + 1u == p.nh) {
        for (uint col = q_dim + tid; col < p.padded_qdim; col += 256u) {
            rot_m[col] = 0.0f;
            rot_hm[col] = half(0.0h);
        }
    }
}

// Single-Launch Hazard-Free Fused Prep + Flash-Decoding + Gated-Q + FWHT32 for `batch_m == 1` (`hd <= 256`)
[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_gqa_fused_m1(
    device const float* qkv_gqa [[buffer(0)]],
    device const float* q_norm  [[buffer(1)]],
    device const float* k_norm  [[buffer(2)]],
    device const float* q_bias  [[buffer(3)]],
    device const float* k_bias  [[buffer(4)]],
    device const float* v_bias  [[buffer(5)]],
    device half* k_cache        [[buffer(6)]],
    device half* v_cache        [[buffer(7)]],
    device const float* kv_gqa  [[buffer(8)]],
    device float* x_rot_out     [[buffer(9)]],
    constant TurboGqaParams& p  [[buffer(10)]],
    device half* x_rot_h_out    [[buffer(11)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint3 tg_pos                [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    if (h >= p.nh) return;

    uint hd = p.hd;
    uint pos = p.pos_base;
    uint q_dim = p.nh * hd;
    uint group_ratio = p.nh / p.nkv;
    uint kv_h = h / group_ratio;
    bool is_kv_writer = (h == kv_h * group_ratio);
    ulong kv_stride = ulong(p.nkv * hd);
    float scale = rsqrt(float(hd));

    bool gated_q = (p.flags & 1u) != 0u;
    bool neox = (p.flags & 2u) != 0u;
    bool has_qn = (p.flags & 4u) != 0u;
    bool has_kn = (p.flags & 8u) != 0u;
    bool has_qb = (p.flags & 16u) != 0u;
    bool has_kb = (p.flags & 32u) != 0u;
    bool has_vb = (p.flags & 64u) != 0u;

    threadgroup float q_sh[256];
    threadgroup float k_tmp[256];
    threadgroup half k_now[256];
    threadgroup half v_now[256];
    threadgroup float sg_m[8];
    threadgroup float sg_l[8];
    threadgroup float sg_acc[8 * 256];

    uint rdim = min(p.rope_dim, hd);
    uint half_r = rdim >> 1u;
    float inv_rdim_log2 = (rdim > 0u) ? (-2.0f * log2(p.rope_freq_base) / float(rdim)) : 0.0f;

    if (sg_id == 0u) {
        uint q_base = gated_q ? (h * 2u * hd) : (h * hd);
        for (uint d = lane; d < hd; d += 32u) {
            q_sh[d] = qkv_gqa[q_base + d] + (has_qb ? q_bias[q_base + d] : 0.0f);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (has_qn) {
            float sq = 0.0f;
            for (uint d = lane; d < hd; d += 32u) {
                float v = q_sh[d];
                sq = fma(v, v, sq);
            }
            sq = simd_sum(sq);
            float sc = rsqrt(sq / float(hd) + p.eps);
            for (uint d = lane; d < hd; d += 32u) q_sh[d] *= sc * q_norm[d];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = lane; i < half_r; i += 32u) {
            float theta = float(pos) * exp2(float(i) * inv_rdim_log2);
            float cs = cos(theta);
            float sn = sin(theta);
            uint i0 = neox ? i : (2u * i);
            uint i1 = neox ? (i + half_r) : (2u * i + 1u);
            float x0 = q_sh[i0];
            float x1 = q_sh[i1];
            q_sh[i0] = x0 * cs - x1 * sn;
            q_sh[i1] = x0 * sn + x1 * cs;
        }
    } else if (sg_id == 1u) {
        device const float* k_cur = kv_gqa;
        device const float* v_cur = k_cur + p.nkv * hd;
        uint k_base = kv_h * hd;
        ulong cache_off = ulong(pos) * kv_stride + ulong(k_base);
        for (uint d = lane; d < hd; d += 32u) {
            k_tmp[d] = k_cur[k_base + d] + (has_kb ? k_bias[k_base + d] : 0.0f);
            half vh_d = half(v_cur[k_base + d] + (has_vb ? v_bias[k_base + d] : 0.0f));
            v_now[d] = vh_d;
            if (is_kv_writer) v_cache[cache_off + d] = vh_d;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (has_kn) {
            float sq = 0.0f;
            for (uint d = lane; d < hd; d += 32u) {
                float v = k_tmp[d];
                sq = fma(v, v, sq);
            }
            sq = simd_sum(sq);
            float sc = rsqrt(sq / float(hd) + p.eps);
            for (uint d = lane; d < hd; d += 32u) k_tmp[d] *= sc * k_norm[d];
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = lane; i < half_r; i += 32u) {
            float theta = float(pos) * exp2(float(i) * inv_rdim_log2);
            float cs = cos(theta);
            float sn = sin(theta);
            uint i0 = neox ? i : (2u * i);
            uint i1 = neox ? (i + half_r) : (2u * i + 1u);
            float x0 = k_tmp[i0];
            float x1 = k_tmp[i1];
            k_tmp[i0] = x0 * cs - x1 * sn;
            k_tmp[i1] = x0 * sn + x1 * cs;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = lane; d < hd; d += 32u) {
            half kh_d = half(k_tmp[d]);
            k_now[d] = kh_d;
            if (is_kv_writer) k_cache[cache_off + d] = kh_d;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i = -INFINITY;
    float l_i = 0.0f;

    if (hd == 256u) {
        uint d4 = lane << 2u;
        float4 qf0 = *(threadgroup const float4*)(q_sh + d4);
        float4 qf1 = *(threadgroup const float4*)(q_sh + 128u + d4);
        float4 acc0 = float4(0.0f);
        float4 acc1 = float4(0.0f);
        for (uint t = sg_id; t < pos; t += 8u) {
            ulong base = ulong(t) * kv_stride + ulong(kv_h * 256u);
            device const half4* kh4 = (device const half4*)(k_cache + base);
            device const half4* vh4 = (device const half4*)(v_cache + base);
            half4 k0 = kh4[lane], k1 = kh4[lane + 32u];
            float dot_v = dot(qf0, float4(k0)) + dot(qf1, float4(k1));
            float s = simd_sum(dot_v) * scale;
            float m_new = max(m_i, s);
            float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
            float w = exp(s - m_new);
            l_i = fma(l_i, alpha, w);
            m_i = m_new;
            half4 v0 = vh4[lane], v1 = vh4[lane + 32u];
            acc0 = fma(float4(w), float4(v0), acc0 * alpha);
            acc1 = fma(float4(w), float4(v1), acc1 * alpha);
        }
        if (sg_id == (pos & 7u)) {
            threadgroup const half4* kh4 = (threadgroup const half4*)k_now;
            threadgroup const half4* vh4 = (threadgroup const half4*)v_now;
            half4 k0 = kh4[lane], k1 = kh4[lane + 32u];
            float dot_v = dot(qf0, float4(k0)) + dot(qf1, float4(k1));
            float s = simd_sum(dot_v) * scale;
            float m_new = max(m_i, s);
            float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
            float w = exp(s - m_new);
            l_i = fma(l_i, alpha, w);
            m_i = m_new;
            half4 v0 = vh4[lane], v1 = vh4[lane + 32u];
            acc0 = fma(float4(w), float4(v0), acc0 * alpha);
            acc1 = fma(float4(w), float4(v1), acc1 * alpha);
        }
        *(threadgroup float4*)(sg_acc + sg_id * 256u + d4) = acc0;
        *(threadgroup float4*)(sg_acc + sg_id * 256u + 128u + d4) = acc1;
    } else if (hd == 128u) {
        uint d4 = lane << 2u;
        float4 qf0 = *(threadgroup const float4*)(q_sh + d4);
        float4 acc0 = float4(0.0f);
        for (uint t = sg_id; t < pos; t += 8u) {
            ulong base = ulong(t) * kv_stride + ulong(kv_h * 128u);
            device const half4* kh4 = (device const half4*)(k_cache + base);
            device const half4* vh4 = (device const half4*)(v_cache + base);
            float dot_v = dot(qf0, float4(kh4[lane]));
            float s = simd_sum(dot_v) * scale;
            float m_new = max(m_i, s);
            float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
            float w = exp(s - m_new);
            l_i = fma(l_i, alpha, w);
            m_i = m_new;
            acc0 = fma(float4(w), float4(vh4[lane]), acc0 * alpha);
        }
        if (sg_id == (pos & 7u)) {
            threadgroup const half4* kh4 = (threadgroup const half4*)k_now;
            threadgroup const half4* vh4 = (threadgroup const half4*)v_now;
            float dot_v = dot(qf0, float4(kh4[lane]));
            float s = simd_sum(dot_v) * scale;
            float m_new = max(m_i, s);
            float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
            float w = exp(s - m_new);
            l_i = fma(l_i, alpha, w);
            m_i = m_new;
            acc0 = fma(float4(w), float4(vh4[lane]), acc0 * alpha);
        }
        *(threadgroup float4*)(sg_acc + sg_id * 256u + d4) = acc0;
    } else {
        uint hd_chunks = (hd + 31u) >> 5u;
        float acc[8];
        for (uint c = 0u; c < hd_chunks; ++c) acc[c] = 0.0f;
        for (uint t = sg_id; t <= pos; t += 8u) {
            bool is_cur = (t == pos);
            ulong base = ulong(t) * kv_stride + ulong(kv_h * hd);
            device const half* kh_g = k_cache + base;
            device const half* vh_g = v_cache + base;
            float dot_v = 0.0f;
            for (uint c = 0u; c < hd_chunks; ++c) {
                uint d = (c << 5u) + lane;
                if (d < hd) dot_v = fma(q_sh[d], float(is_cur ? k_now[d] : kh_g[d]), dot_v);
            }
            float s = simd_sum(dot_v) * scale;
            float m_new = max(m_i, s);
            float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
            float w = exp(s - m_new);
            l_i = fma(l_i, alpha, w);
            m_i = m_new;
            for (uint c = 0u; c < hd_chunks; ++c) {
                uint d = (c << 5u) + lane;
                if (d < hd) acc[c] = fma(w, float(is_cur ? v_now[d] : vh_g[d]), acc[c] * alpha);
            }
        }
        for (uint c = 0u; c < hd_chunks; ++c) {
            uint d = (c << 5u) + lane;
            if (d < hd) sg_acc[sg_id * 256u + d] = acc[c];
        }
    }

    if (lane == 0u) {
        sg_m[sg_id] = m_i;
        sg_l[sg_id] = l_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_all = -INFINITY;
    for (uint sg = 0u; sg < 8u; ++sg) {
        if (sg_l[sg] > 0.0f) m_all = max(m_all, sg_m[sg]);
    }
    float w_sg[8];
    float l_all = 0.0f;
    for (uint sg = 0u; sg < 8u; ++sg) {
        float w = (sg_l[sg] > 0.0f) ? exp(sg_m[sg] - m_all) : 0.0f;
        w_sg[sg] = w;
        l_all = fma(sg_l[sg], w, l_all);
    }
    float inv_l = 1.0f / l_all;

    uint g_base = h * 2u * hd + hd;
    uint hd_blks = hd >> 5u;
    for (uint b = sg_id; b < hd_blks; b += 8u) {
        uint d = (b << 5u) + lane;
        float out_d = 0.0f;
        for (uint sg = 0u; sg < 8u; ++sg) {
            out_d = fma(sg_acc[sg * 256u + d], w_sg[sg], out_d);
        }
        out_d *= inv_l;
        if (gated_q) {
            float g = qkv_gqa[g_base + d] + (has_qb ? q_bias[g_base + d] : 0.0f);
            out_d *= 1.0f / (1.0f + exp(-g));
        }
        uint col = h * hd + d;
        float rot = vxt_fwht32_lane(vxt_apply_col_sign(out_d, col), lane);
        x_rot_h_out[col] = half(rot * 0.0625f);
    }
    if (h + 1u == p.nh) {
        for (uint col = q_dim + tid; col < p.padded_qdim; col += 256u) {
            x_rot_h_out[col] = half(0.0h);
        }
    }
    (void)x_rot_out;
}

// 10. All-GPU 2-Stage Argmax over `logits_buf` (M = 1..4) -> writes `out_ids` & `tok_ring[next_slot]`
inline float vxt_spec_pen_cnt(
    device const uchar* seen_vocab,
    uint r,
    uint m,
    uint d1,
    uint d2,
    uint d3
) {
    uint c = uint(seen_vocab[r]);
    if (m >= 1u && r == d1) ++c;
    if (m >= 2u && r == d2) ++c;
    if (m >= 3u && r == d3) ++c;
    return (c >= 2u) ? (0.35f * float(min(c - 1u, 5u))) : 0.0f;
}

kernel void vxt_argmax_stage1(
    device const float* logits      [[buffer(0)]],
    device float* blk_vals          [[buffer(1)]],
    device uint* blk_idxs           [[buffer(2)]],
    constant TurboArgmaxParams& p   [[buffer(3)]],
    device const uchar* seen_vocab  [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint blk = tg_pos.x;
    uint m = tg_pos.y;
    if (blk >= p.n_blocks || m >= p.batch_m) return;

    device const float* lm = logits + ulong(m) * ulong(p.n_vocab);
    uint base = blk * 1024u;
    float best_v = -INFINITY;
    uint best_i = 0u;
    for (uint k = 0u; k < 4u; ++k) {
        uint idx = base + k * 256u + tid;
        if (idx < p.n_vocab) {
            float v = lm[idx];
            float scnt = (p.presence_pen > 0.0f) ? vxt_spec_pen_cnt(seen_vocab, idx, m, p.d1, p.d2, p.d3) : 0.0f;
            v -= p.presence_pen * scnt;
            if (v > best_v) {
                best_v = v;
                best_i = idx;
            }
        }
    }

    for (uint off = 16u; off > 0u; off >>= 1u) {
        float ov = simd_shuffle_down(best_v, ushort(off));
        uint oi = simd_shuffle_down(best_i, ushort(off));
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    threadgroup float sg_v[8];
    threadgroup uint sg_i[8];
    if (lane == 0u) {
        sg_v[sg_id] = best_v;
        sg_i[sg_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float v = (lane < 8u) ? sg_v[lane] : -INFINITY;
        uint idx = (lane < 8u) ? sg_i[lane] : 0u;
        for (uint off = 4u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_down(v, ushort(off));
            uint oi = simd_shuffle_down(idx, ushort(off));
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0u) {
            blk_vals[m * p.n_blocks + blk] = v;
            blk_idxs[m * p.n_blocks + blk] = idx;
        }
    }
}

kernel void vxt_argmax_stage2(
    device const float* blk_vals    [[buffer(0)]],
    device const uint* blk_idxs     [[buffer(1)]],
    device uint* out_ids            [[buffer(2)]],
    device uint* tok_ring           [[buffer(3)]],
    constant TurboArgmaxParams& p   [[buffer(4)]],
    device uchar* seen_vocab        [[buffer(5)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    if (m >= p.batch_m) return;

    float best_v = -INFINITY;
    uint best_i = 0u;
    for (uint b = tid; b < p.n_blocks; b += 256u) {
        float v = blk_vals[m * p.n_blocks + b];
        uint idx = blk_idxs[m * p.n_blocks + b];
        if (v > best_v || (v == best_v && idx < best_i)) {
            best_v = v;
            best_i = idx;
        }
    }

    for (uint off = 16u; off > 0u; off >>= 1u) {
        float ov = simd_shuffle_down(best_v, ushort(off));
        uint oi = simd_shuffle_down(best_i, ushort(off));
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    threadgroup float sg_v[8];
    threadgroup uint sg_i[8];
    if (lane == 0u) {
        sg_v[sg_id] = best_v;
        sg_i[sg_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float v = (lane < 8u) ? sg_v[lane] : -INFINITY;
        uint idx = (lane < 8u) ? sg_i[lane] : 0u;
        for (uint off = 4u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_down(v, ushort(off));
            uint oi = simd_shuffle_down(idx, ushort(off));
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0u) {
            out_ids[m] = idx;
            if (m == 0u && p.write_next_slot != 0xffffffffu) {
                tok_ring[p.write_next_slot] = idx;
            }
            if (p.batch_m == 1u && p.presence_pen > 0.0f && idx < p.n_vocab) {
                uchar c = seen_vocab[idx];
                if (c < 255u) seen_vocab[idx] = c + 1u;
            }
        }
    }
}

// 11. 2-Bit Individual Top-Coordinate (`ushort` per `(col, tile8)`) LM Head Sketch + Exact Stage 2
inline void vxt_select_top_coords_m1(
    device const float* xm,
    device uint4* active_grp4,
    device uint* qmeta_out,
    threadgroup float* sg_tmp,
    uint padded_cols,
    uint tid,
    uint sg_id,
    uint lane
) {
    float max_a = 0.0f;
    for (uint col = tid; col < padded_cols; col += 256u) {
        max_a = max(max_a, abs(xm[col]));
    }
    max_a = simd_max(max_a);
    if (lane == 0u) sg_tmp[sg_id] = max_a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float m = (lane < 8u) ? sg_tmp[lane] : 0.0f;
        m = simd_max(m);
        if (lane == 0u) sg_tmp[0] = (m > 1e-20f) ? m : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float glob_max = sg_tmp[0];
    float inv_max = 1.0f / glob_max;
    if (tid == 0u) {
        qmeta_out[0] = as_type<uint>(glob_max);
    }

    uint steps = padded_cols >> 8u;
    uint k_ind = (steps <= 2u) ? 256u : ((steps <= 8u) ? 56u : min(64u, max(24u, 448u / steps)));
    device uint2* active_ind = (device uint2*)active_grp4;

    for (uint step = sg_id; step < steps; step += 8u) {
        uint c_base = (step << 8u) + (lane << 3u);
        float4 v0 = (*(device const float4*)(xm + c_base)) * inv_max;
        float4 v1 = (*(device const float4*)(xm + c_base + 4u)) * inv_max;
        float vv[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };

        if (k_ind >= 256u) {
            device uint2* dst = active_ind + (step << 8u);
            for (uint j = 0u; j < 8u; ++j) {
                uint xh = uint(as_type<ushort>(half(vv[j])));
                dst[(lane << 3u) + j] = uint2(xh | (xh << 16u), c_base + j);
            }
            if (lane == 0u) qmeta_out[1u + step] = 256u;
            continue;
        }

        float aa[8];
        float lmax = 0.0f;
        for (uint j = 0u; j < 8u; ++j) {
            aa[j] = abs(vv[j]);
            lmax = max(lmax, aa[j]);
        }
        float hi = simd_max(lmax);
        if (hi <= 1e-6f) {
            if (lane == 0u) qmeta_out[1u + step] = 0u;
            continue;
        }
        float lo = 0.0f;
        for (uint it = 0u; it < 10u; ++it) {
            float mid = 0.5f * (lo + hi);
            uint cnt = 0u;
            for (uint j = 0u; j < 8u; ++j) cnt += uint(aa[j] > mid);
            if (simd_sum(cnt) > k_ind) lo = mid;
            else hi = mid;
        }

        uint scnt = 0u, tcnt = 0u;
        for (uint j = 0u; j < 8u; ++j) {
            if (aa[j] > hi) ++scnt;
            else if (lo > 0.0f && aa[j] >= lo) ++tcnt;
        }
        uint stot = simd_sum(scnt);
        uint tie_need = k_ind - stot;
        uint trank = simd_prefix_exclusive_sum(tcnt);

        bool sel[8];
        uint mcnt = 0u;
        for (uint j = 0u; j < 8u; ++j) {
            bool sj = (aa[j] > hi);
            if (!sj && lo > 0.0f && aa[j] >= lo) {
                if (trank < tie_need) sj = true;
                ++trank;
            }
            sel[j] = sj;
            mcnt += uint(sj);
        }

        uint pos = simd_prefix_exclusive_sum(mcnt);
        device uint2* dst = active_ind + (step << 8u);
        for (uint j = 0u; j < 8u; ++j) {
            if (sel[j]) {
                uint xh = uint(as_type<ushort>(half(vv[j])));
                dst[pos++] = uint2(xh | (xh << 16u), c_base + j);
            }
        }
        uint final_tot = simd_sum(mcnt);
        if (lane == 0u) {
            qmeta_out[1u + step] = final_tot;
        }
    }
}

kernel void vxt_build_lm_sketch(
    device const ushort* scales_f16     [[buffer(0)]],
    device const uint* packed_words     [[buffer(1)]],
    device ushort* sketch_signs         [[buffer(2)]],
    device uint4* sketch_l1_u4          [[buffer(3)]],
    constant TurboSketchBuildParams& p  [[buffer(4)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint tile8 = tg_pos.x;
    if (tile8 >= p.stride_tiles8) return;

    uint steps = p.padded_cols >> 8u;
    uint r = (tile8 << 3u) + sg_id;
    threadgroup uchar sg_s8[8][32];
    threadgroup uchar sg_b8[8][32];
    threadgroup half sg_l1_h[8];

    for (uint step = 0u; step < steps; ++step) {
        uint s8 = 0u;
        float mag[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        if (r < p.rows) {
            uint tile = r >> 2u;
            uint sr = r & 3u;
            float sc0 = float(as_type<half>(scales_f16[(ulong(tile) * ulong(steps) + ulong(step)) * 8ul + ulong(sr)]));
            float sc1 = float(as_type<half>(scales_f16[(ulong(tile) * ulong(steps) + ulong(step)) * 8ul + ulong(4u + sr)]));
            float sc = (lane < 16u) ? sc0 : sc1;
            uint nz8 = 0u;
            if (p.bits == 2u) {
                device const uint* p64 = packed_words + (ulong(tile) * ulong(steps) + ulong(step)) * 64ul;
                uint w = p64[lane * 2u + (sr >> 1u)];
                uint c16 = (w >> ((sr & 1u) * 16u)) & 0xffffu;
                uint b0 = c16 & 0xffu;
                uint b1 = c16 >> 8u;
                s8 = ((b0 >> 4u) & 0x0fu) | (b1 & 0xf0u);
                float4 m0 = abs(float4(vxt_orth4d_const(b0 & 15u, ORTH4D_16_U16)));
                float4 m1 = abs(float4(vxt_orth4d_const(b1 & 15u, ORTH4D_16_U16)));
                mag[0] = sc * m0.x; mag[1] = sc * m0.y; mag[2] = sc * m0.z; mag[3] = sc * m0.w;
                mag[4] = sc * m1.x; mag[5] = sc * m1.y; mag[6] = sc * m1.z; mag[7] = sc * m1.w;
                if (sc > 0.0f) nz8 = 0xffu;
            } else if (p.bits == 5u) {
                device const uint* p80 = packed_words + (ulong(tile) * ulong(steps) + ulong(step)) * 80ul;
                uint2 cc = vxt_t25_unpack_row(p80, lane, sr);
                s8 = (cc.x >> 6u) | ((cc.y >> 6u) << 4u);
                float4 m0 = abs(float4(vxt_orth4d64_const(cc.x & 63u)));
                float4 m1 = abs(float4(vxt_orth4d64_const(cc.y & 63u)));
                mag[0] = sc * m0.x; mag[1] = sc * m0.y; mag[2] = sc * m0.z; mag[3] = sc * m0.w;
                mag[4] = sc * m1.x; mag[5] = sc * m1.y; mag[6] = sc * m1.z; mag[7] = sc * m1.w;
                if (sc > 0.0f) {
                    nz8 = ((m0.x > 0.0f) ? 1u : 0u) | ((m0.y > 0.0f) ? 2u : 0u) | ((m0.z > 0.0f) ? 4u : 0u) | ((m0.w > 0.0f) ? 8u : 0u)
                        | ((m1.x > 0.0f) ? 16u : 0u) | ((m1.y > 0.0f) ? 32u : 0u) | ((m1.z > 0.0f) ? 64u : 0u) | ((m1.w > 0.0f) ? 128u : 0u);
                }
            } else {
                device const uint* p96 = packed_words + (ulong(tile) * ulong(steps) + ulong(step)) * 96ul;
                uint rw0 = p96[lane * 2u], rw1 = p96[lane * 2u + 1u], rw2 = p96[64u + lane];
                bool is_ec = (p.bits == 4u);
                for (uint pr = 0u; pr < 4u; ++pr) {
                    uint code6 = vxt_t3_unpack_code6(rw0, rw1, rw2, sr, pr);
                    s8 |= ((code6 >> 4u) & 3u) << (pr * 2u);
                    float2 m2 = abs(float2(vxt_orth2d_const_sel(code6 & 15u, is_ec)));
                    mag[2u * pr] = sc * m2.x;
                    mag[2u * pr + 1u] = sc * m2.y;
                    if (sc > 0.0f) {
                        uint pair_nz = ((m2.x > 0.0f) ? 1u : 0u) | ((m2.y > 0.0f) ? 2u : 0u);
                        nz8 |= pair_nz << (pr * 2u);
                    }
                }
            }
            uint h_rnd = (r * 0x9e3779b9u) ^ (step * 0x85ebca6bu) ^ (lane * 0xc2b2ae35u);
            h_rnd ^= h_rnd >> 15u;
            h_rnd *= 0x2c1b3c6du;
            uint rnd8 = (h_rnd >> 16u) & 0xffu;
            s8 = (s8 & nz8) | (rnd8 & ~nz8);
        }

        float lane_e2 = 0.0f;
        for (uint j = 0u; j < 8u; ++j) lane_e2 = fma(mag[j], mag[j], lane_e2);
        float rms_step = sqrt(simd_sum(lane_e2) * (1.0f / 256.0f));
        float thr = 0.98f * rms_step;

        uint big8 = 0u;
        float lane_wc = 0.0f;
        float lane_cc = 0.0f;
        for (uint j = 0u; j < 8u; ++j) {
            bool is_big = (mag[j] > thr);
            if (is_big) big8 |= (1u << j);
            float cm = is_big ? 1.5f : 0.5f;
            lane_wc = fma(mag[j], cm, lane_wc);
            lane_cc = fma(cm, cm, lane_cc);
        }
        float step_wc = simd_sum(lane_wc);
        float step_cc = simd_sum(lane_cc);
        if (lane == 0u) {
            sg_l1_h[sg_id] = (r < p.rows && step_cc > 0.0f) ? half(step_wc / step_cc) : half(0.0h);
        }
        sg_s8[sg_id][lane] = uchar(s8);
        sg_b8[sg_id][lane] = uchar(big8);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint lane_src = tid >> 3u;
        uint bit_idx = tid & 7u;
        uint u16 = 0u;
        for (uint m = 0u; m < 4u; ++m) {
            uint se = (uint(sg_s8[2u * m][lane_src]) >> bit_idx) & 1u;
            uint be = (uint(sg_b8[2u * m][lane_src]) >> bit_idx) & 1u;
            uint so = (uint(sg_s8[2u * m + 1u][lane_src]) >> bit_idx) & 1u;
            uint bo = (uint(sg_b8[2u * m + 1u][lane_src]) >> bit_idx) & 1u;
            u16 |= (se << (7u - m)) | (be << (3u - m)) | (so << (15u - m)) | (bo << (11u - m));
        }
        ulong col_idx = (ulong(step) << 8u) + ulong(tid);
        sketch_signs[col_idx * ulong(p.stride_tiles8) + ulong(tile8)] = ushort(u16);
        if (tid == 0u) {
            sketch_l1_u4[ulong(step) * ulong(p.stride_tiles8) + ulong(tile8)] = *(threadgroup const uint4*)sg_l1_h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void vxt_quantize_x_sketch(
    device const float* x_rot           [[buffer(0)]],
    device uint4* active_grp4_out       [[buffer(1)]],
    device uint* x_qmeta_out            [[buffer(2)]],
    constant uint2& qp                  [[buffer(3)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    if (m >= qp.y) return;
    uint steps = qp.x >> 8u;
    threadgroup float sg_tmp[8];
    vxt_select_top_coords_m1(
        x_rot + ulong(m) * ulong(qp.x),
        active_grp4_out + ulong(m) * ulong(steps * 128u),
        x_qmeta_out + ulong(m) * ulong(1u + steps),
        sg_tmp,
        qp.x,
        tid,
        sg_id,
        lane
    );
}

[[max_total_threads_per_threadgroup(128)]]
kernel void vxt_lm_sketch_stage1(
    device const ushort* sketch_signs   [[buffer(0)]],
    device const uint4* sketch_l1_u4    [[buffer(1)]],
    device const uint4* active_grp4_in  [[buffer(2)]],
    device const uint* x_qmeta_in       [[buffer(3)]],
    device float* blk_sketch_max_out    [[buffer(4)]],
    constant TurboSketchEvalParams& p   [[buffer(5)]],
    device const uchar* seen_vocab      [[buffer(6)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    uint tile8 = tg_pos.x * 128u + tid;
    uint steps = p.padded_cols >> 8u;
    float4 facc0 = float4(0.0f);
    float4 facc1 = float4(0.0f);
    device const uint4* active_grp4 = active_grp4_in + ulong(m) * ulong(steps * 128u);
    device const uint* x_qmeta = x_qmeta_in + ulong(m) * ulong(1u + steps);
    device float* blk_sketch_max = blk_sketch_max_out + ulong(m) * ulong(p.n_blocks);
    device const uint2* active_ind = (device const uint2*)active_grp4;

    threadgroup uint sh_cnt[8];
    threadgroup uint sh_col[512];
    threadgroup uint sh_xh2[512];

    if (steps > 2u && steps <= 8u) {
        if (tid < steps) {
            sh_cnt[tid] = min(x_qmeta[1u + tid], 64u);
        }
        for (uint idx = tid; idx < (steps << 6u); idx += 128u) {
            uint s_ld = idx >> 6u;
            uint k_ld = idx & 63u;
            uint2 ai = active_ind[(s_ld << 8u) + k_ld];
            sh_xh2[idx] = ai.x;
            sh_col[idx] = ai.y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tile8 >= p.stride_tiles8) return;
        for (uint step = 0u; step < steps; ++step) {
            uint k_cnt = sh_cnt[step];
            uint base = step << 6u;
            half2 h0 = half2(0.0h), h1 = half2(0.0h), h2 = half2(0.0h), h3 = half2(0.0h);
            for (uint k = 0u; k < k_cnt; ++k) {
                uint col = sh_col[base + k];
                half2 x_h2 = as_type<half2>(sh_xh2[base + k]);
                uint u16 = uint(sketch_signs[ulong(col) * ulong(p.stride_tiles8) + ulong(tile8)]);
                uint w32 = (u16 & 0x00ffu) | ((u16 & 0xff00u) << 8u);
                uint sb0 = (w32 <<  8u) & 0x80008000u, mb0 = (w32 <<  7u) & 0x04000400u;
                uint sb1 = (w32 <<  9u) & 0x80008000u, mb1 = (w32 <<  8u) & 0x04000400u;
                uint sb2 = (w32 << 10u) & 0x80008000u, mb2 = (w32 <<  9u) & 0x04000400u;
                uint sb3 = (w32 << 11u) & 0x80008000u, mb3 = (w32 << 10u) & 0x04000400u;
                h0 = fma(as_type<half2>(0x38003800u ^ sb0 ^ mb0 ^ (mb0 >> 1u)), x_h2, h0);
                h1 = fma(as_type<half2>(0x38003800u ^ sb1 ^ mb1 ^ (mb1 >> 1u)), x_h2, h1);
                h2 = fma(as_type<half2>(0x38003800u ^ sb2 ^ mb2 ^ (mb2 >> 1u)), x_h2, h2);
                h3 = fma(as_type<half2>(0x38003800u ^ sb3 ^ mb3 ^ (mb3 >> 1u)), x_h2, h3);
            }
            uint4 lu = sketch_l1_u4[ulong(step) * ulong(p.stride_tiles8) + ulong(tile8)];
            facc0 = fma(float4(as_type<half4>(lu.xy)), float4(half4(h0, h1)), facc0);
            facc1 = fma(float4(as_type<half4>(lu.zw)), float4(half4(h2, h3)), facc1);
        }
    } else {
        for (uint step = 0u; step < steps; ++step) {
            if (tid == 0u) sh_cnt[0] = min(x_qmeta[1u + step], 256u);
            for (uint idx = tid; idx < 256u; idx += 128u) {
                uint2 ai = active_ind[(step << 8u) + idx];
                sh_xh2[idx] = ai.x;
                sh_col[idx] = ai.y;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tile8 < p.stride_tiles8) {
                uint k_cnt = sh_cnt[0];
                half2 h0 = half2(0.0h), h1 = half2(0.0h), h2 = half2(0.0h), h3 = half2(0.0h);
                for (uint k = 0u; k < k_cnt; ++k) {
                    uint col = sh_col[k];
                    half2 x_h2 = as_type<half2>(sh_xh2[k]);
                    uint u16 = uint(sketch_signs[ulong(col) * ulong(p.stride_tiles8) + ulong(tile8)]);
                    uint w32 = (u16 & 0x00ffu) | ((u16 & 0xff00u) << 8u);
                    uint sb0 = (w32 <<  8u) & 0x80008000u, mb0 = (w32 <<  7u) & 0x04000400u;
                    uint sb1 = (w32 <<  9u) & 0x80008000u, mb1 = (w32 <<  8u) & 0x04000400u;
                    uint sb2 = (w32 << 10u) & 0x80008000u, mb2 = (w32 <<  9u) & 0x04000400u;
                    uint sb3 = (w32 << 11u) & 0x80008000u, mb3 = (w32 << 10u) & 0x04000400u;
                    h0 = fma(as_type<half2>(0x38003800u ^ sb0 ^ mb0 ^ (mb0 >> 1u)), x_h2, h0);
                    h1 = fma(as_type<half2>(0x38003800u ^ sb1 ^ mb1 ^ (mb1 >> 1u)), x_h2, h1);
                    h2 = fma(as_type<half2>(0x38003800u ^ sb2 ^ mb2 ^ (mb2 >> 1u)), x_h2, h2);
                    h3 = fma(as_type<half2>(0x38003800u ^ sb3 ^ mb3 ^ (mb3 >> 1u)), x_h2, h3);
                }
                uint4 lu = sketch_l1_u4[ulong(step) * ulong(p.stride_tiles8) + ulong(tile8)];
                facc0 = fma(float4(as_type<half4>(lu.xy)), float4(half4(h0, h1)), facc0);
                facc1 = fma(float4(as_type<half4>(lu.zw)), float4(half4(h2, h3)), facc1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tile8 >= p.stride_tiles8) return;
    }

    float glob_max = as_type<float>(x_qmeta[0]);
    uint r_base = tile8 << 3u;
    float t_max = -INFINITY;
    if (r_base < p.rows) {
        float pen = p.presence_pen;
        for (uint k = 0u; k < 4u; ++k) {
            uint r0 = r_base + k;
            if (r0 < p.rows) {
                float c0 = (pen > 0.0f) ? vxt_spec_pen_cnt(seen_vocab, r0, m, p.d1, p.d2, p.d3) : 0.0f;
                float v0 = facc0[k] * glob_max - pen * c0;
                t_max = max(t_max, v0);
            }
            uint r1 = r_base + 4u + k;
            if (r1 < p.rows) {
                float c1 = (pen > 0.0f) ? vxt_spec_pen_cnt(seen_vocab, r1, m, p.d1, p.d2, p.d3) : 0.0f;
                float v1 = facc1[k] * glob_max - pen * c1;
                t_max = max(t_max, v1);
            }
        }
    }

    t_max = max(t_max, simd_shuffle_xor(t_max, 1u));
    t_max = max(t_max, simd_shuffle_xor(t_max, 2u));
    t_max = max(t_max, simd_shuffle_xor(t_max, 4u));
    uint block_id = tile8 >> 3u;
    if ((lane & 7u) == 0u && block_id < p.n_blocks) {
        blk_sketch_max[block_id] = t_max;
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_lm_sketch_threshold(
    device const float* blk_sketch_max_in [[buffer(0)]],
    device uint* surv_out_all             [[buffer(1)]],
    constant uint2& tp                    [[buffer(2)]],
    uint tid                              [[thread_index_in_threadgroup]],
    uint sg_id                            [[simdgroup_index_in_threadgroup]],
    uint lane                             [[thread_index_in_simdgroup]],
    uint3 tg_pos                          [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    if (m >= tp.y) return;
    uint n_blocks = tp.x;
    device const float* blk_sketch_max = blk_sketch_max_in + ulong(m) * ulong(n_blocks);
    device uint* surv_out = surv_out_all + ulong(m) * 512ul;
    const uint MAX_SURV = 256u;
    if (n_blocks <= MAX_SURV) {
        if (tid == 0u) surv_out[0] = n_blocks;
        if (tid < n_blocks) surv_out[1u + tid] = tid;
        return;
    }

    float rv[16];
    bool use_rv = (n_blocks <= 4096u);
    float lo_t = INFINITY;
    float hi_t = -INFINITY;
    if (use_rv) {
        for (uint k = 0u; k < 16u; ++k) {
            uint b = (k << 8u) + tid;
            float v = (b < n_blocks) ? blk_sketch_max[b] : -INFINITY;
            rv[k] = v;
            if (b < n_blocks) {
                lo_t = min(lo_t, v);
                hi_t = max(hi_t, v);
            }
        }
    } else {
        for (uint b = tid; b < n_blocks; b += 256u) {
            float v = blk_sketch_max[b];
            lo_t = min(lo_t, v);
            hi_t = max(hi_t, v);
        }
    }
    lo_t = simd_min(lo_t);
    hi_t = simd_max(hi_t);

    threadgroup float sg_lo[8];
    threadgroup float sg_hi[8];
    threadgroup uint sg_cnt[8];
    if (lane == 0u) {
        sg_lo[sg_id] = lo_t;
        sg_hi[sg_id] = hi_t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float l = (lane < 8u) ? sg_lo[lane] : INFINITY;
        float h = (lane < 8u) ? sg_hi[lane] : -INFINITY;
        l = simd_min(l);
        h = simd_max(h);
        if (lane == 0u) {
            sg_lo[0] = l;
            sg_hi[0] = h;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lo = sg_lo[0];
    float hi = sg_hi[0];
    for (uint it = 0u; it < 9u; ++it) {
        float mid = 0.5f * (lo + hi);
        uint c = 0u;
        if (use_rv) {
            for (uint k = 0u; k < 16u; ++k) c += uint(rv[k] > mid);
        } else {
            for (uint b = tid; b < n_blocks; b += 256u) {
                if (blk_sketch_max[b] > mid) ++c;
            }
        }
        c = simd_sum(c);
        if (lane == 0u) sg_cnt[sg_id] = c;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u) {
            uint tot = (lane < 8u) ? sg_cnt[lane] : 0u;
            tot = simd_sum(tot);
            if (lane == 0u) sg_cnt[0] = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_cnt[0] > MAX_SURV) lo = mid;
        else hi = mid;
    }

    threadgroup atomic_uint tg_n;
    if (tid == 0u) atomic_store_explicit(&tg_n, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (use_rv) {
        for (uint k = 0u; k < 16u; ++k) {
            if (rv[k] > hi) {
                uint idx = atomic_fetch_add_explicit(&tg_n, 1u, memory_order_relaxed);
                if (idx < MAX_SURV) surv_out[1u + idx] = (k << 8u) + tid;
            }
        }
    } else {
        for (uint b = tid; b < n_blocks; b += 256u) {
            if (blk_sketch_max[b] > hi) {
                uint idx = atomic_fetch_add_explicit(&tg_n, 1u, memory_order_relaxed);
                if (idx < MAX_SURV) surv_out[1u + idx] = b;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (atomic_load_explicit(&tg_n, memory_order_relaxed) < MAX_SURV) {
        if (use_rv) {
            for (uint k = 0u; k < 16u; ++k) {
                float v = rv[k];
                uint b = (k << 8u) + tid;
                if (b < n_blocks && v <= hi && v >= lo) {
                    uint idx = atomic_fetch_add_explicit(&tg_n, 1u, memory_order_relaxed);
                    if (idx < MAX_SURV) surv_out[1u + idx] = b;
                }
            }
        } else {
            for (uint b = tid; b < n_blocks; b += 256u) {
                float v = blk_sketch_max[b];
                if (v <= hi && v >= lo) {
                    uint idx = atomic_fetch_add_explicit(&tg_n, 1u, memory_order_relaxed);
                    if (idx < MAX_SURV) surv_out[1u + idx] = b;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        surv_out[0] = min(atomic_load_explicit(&tg_n, memory_order_relaxed), MAX_SURV);
    }
}

inline void vxt_stage2_reduce_tg(
    float best_v,
    uint best_i,
    device float* blk_vals,
    device uint* blk_idxs,
    threadgroup float* sg_v,
    threadgroup uint* sg_i,
    uint tg_idx,
    uint sg_id,
    uint lane
) {
    if (lane == 0u) {
        sg_v[sg_id] = best_v;
        sg_i[sg_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float v = (lane < 8u) ? sg_v[lane] : -INFINITY;
        uint idx = (lane < 8u) ? sg_i[lane] : 0u;
        for (uint off = 4u; off > 0u; off >>= 1u) {
            float ov = simd_shuffle_down(v, ushort(off));
            uint oi = simd_shuffle_down(idx, ushort(off));
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0u) {
            blk_vals[tg_idx] = v;
            blk_idxs[tg_idx] = idx;
        }
    }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_lm_exact_stage2_turbo2(
    device const ushort* scales_f16     [[buffer(0)]],
    device const uint* packed_words     [[buffer(1)]],
    device const half4* x_rot_h_in      [[buffer(2)]],
    device const uint* surv_buf_in      [[buffer(3)]],
    device float* blk_vals_out          [[buffer(4)]],
    device uint* blk_idxs_out           [[buffer(5)]],
    constant TurboSketchStage2Params& p [[buffer(6)]],
    device const uchar* seen_vocab      [[buffer(7)]],
    device const half4* lut_bank        [[buffer(8)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    device const uint* surv_buf = surv_buf_in + ulong(m) * 512ul;
    device float* blk_vals = blk_vals_out + ulong(m) * ulong(p.stage2_tgs);
    device uint* blk_idxs = blk_idxs_out + ulong(m) * ulong(p.stage2_tgs);
    uint surv_slot = tg_pos.x >> 1u;
    if (surv_slot >= surv_buf[0]) {
        if (tid == 0u) {
            blk_vals[tg_pos.x] = -INFINITY;
            blk_idxs[tg_pos.x] = 0u;
        }
        return;
    }

    threadgroup half4 lut256[256];
    lut256[tid] = lut_bank[1024u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half4* x_rot_h = x_rot_h_in + ulong(m) * ulong(p.padded_cols >> 2u);
    uint block_id = surv_buf[1u + surv_slot];
    uint tile = (block_id << 4u) + ((tg_pos.x & 1u) << 3u) + sg_id;
    uint r0 = tile << 2u;
    float best_v = -INFINITY;
    uint best_i = 0u;

    if (r0 < p.rows) {
        uint steps = p.padded_cols >> 8u;
        uint gh = lane >> 4u;
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint2* pw0 = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);
        float4 acc0 = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            acc0 = vxt_eval_t2_lut(
                lut256, pw0[(step << 5u) + lane],
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
        float4 s0 = simd_sum(acc0) * 16.0f;
        if (lane == 0u) {
            for (uint k = 0u; k < 4u; ++k) {
                uint rk = r0 + k;
                if (rk < p.rows) {
                    float sc = (p.presence_pen > 0.0f) ? vxt_spec_pen_cnt(seen_vocab, rk, m, p.d1, p.d2, p.d3) : 0.0f;
                    float v = s0[k] - p.presence_pen * sc;
                    if (v > best_v || (v == best_v && rk < best_i)) { best_v = v; best_i = rk; }
                }
            }
        }
    }

    threadgroup float sg_v[8];
    threadgroup uint sg_i[8];
    vxt_stage2_reduce_tg(best_v, best_i, blk_vals, blk_idxs, sg_v, sg_i, tg_pos.x, sg_id, lane);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_lm_exact_stage2_turbo25(
    device const ushort* scales_f16     [[buffer(0)]],
    device const uint* packed_words     [[buffer(1)]],
    device const half4* x_rot_h_in      [[buffer(2)]],
    device const uint* surv_buf_in      [[buffer(3)]],
    device float* blk_vals_out          [[buffer(4)]],
    device uint* blk_idxs_out           [[buffer(5)]],
    constant TurboSketchStage2Params& p [[buffer(6)]],
    device const uchar* seen_vocab      [[buffer(7)]],
    device const half4* lut_bank        [[buffer(8)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    device const uint* surv_buf = surv_buf_in + ulong(m) * 512ul;
    device float* blk_vals = blk_vals_out + ulong(m) * ulong(p.stage2_tgs);
    device uint* blk_idxs = blk_idxs_out + ulong(m) * ulong(p.stage2_tgs);
    uint surv_slot = tg_pos.x >> 1u;
    if (surv_slot >= surv_buf[0]) {
        if (tid == 0u) {
            blk_vals[tg_pos.x] = -INFINITY;
            blk_idxs[tg_pos.x] = 0u;
        }
        return;
    }

    threadgroup half4 lut1k[1024];
    for (uint i = tid; i < 1024u; i += 256u) lut1k[i] = lut_bank[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half4* x_rot_h = x_rot_h_in + ulong(m) * ulong(p.padded_cols >> 2u);
    uint block_id = surv_buf[1u + surv_slot];
    uint tile = (block_id << 4u) + ((tg_pos.x & 1u) << 3u) + sg_id;
    uint r0 = tile << 2u;
    float best_v = -INFINITY;
    uint best_i = 0u;

    if (r0 < p.rows) {
        uint steps = p.padded_cols >> 8u;
        uint gh = lane >> 4u;
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 80ul;
        float4 acc0 = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            device const uint* p0 = pw0 + step * 80u;
            acc0 = vxt_eval_t25_lut1k(
                lut1k, ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
        float4 s0 = simd_sum(acc0) * 16.0f;
        if (lane == 0u) {
            for (uint k = 0u; k < 4u; ++k) {
                uint rk = r0 + k;
                if (rk < p.rows) {
                    float sc = (p.presence_pen > 0.0f) ? vxt_spec_pen_cnt(seen_vocab, rk, m, p.d1, p.d2, p.d3) : 0.0f;
                    float v = s0[k] - p.presence_pen * sc;
                    if (v > best_v || (v == best_v && rk < best_i)) { best_v = v; best_i = rk; }
                }
            }
        }
    }

    threadgroup float sg_v[8];
    threadgroup uint sg_i[8];
    vxt_stage2_reduce_tg(best_v, best_i, blk_vals, blk_idxs, sg_v, sg_i, tg_pos.x, sg_id, lane);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void vxt_lm_exact_stage2_turbo3(
    device const ushort* scales_f16     [[buffer(0)]],
    device const uint* packed_words     [[buffer(1)]],
    device const half4* x_rot_h_in      [[buffer(2)]],
    device const uint* surv_buf_in      [[buffer(3)]],
    device float* blk_vals_out          [[buffer(4)]],
    device uint* blk_idxs_out           [[buffer(5)]],
    constant TurboSketchStage2Params& p [[buffer(6)]],
    device const uchar* seen_vocab      [[buffer(7)]],
    device const half4* lut_bank        [[buffer(8)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    device const uint* surv_buf = surv_buf_in + ulong(m) * 512ul;
    device float* blk_vals = blk_vals_out + ulong(m) * ulong(p.stage2_tgs);
    device uint* blk_idxs = blk_idxs_out + ulong(m) * ulong(p.stage2_tgs);
    uint surv_slot = tg_pos.x >> 1u;
    if (surv_slot >= surv_buf[0]) {
        if (tid == 0u) {
            blk_vals[tg_pos.x] = -INFINITY;
            blk_idxs[tg_pos.x] = 0u;
        }
        return;
    }

    bool is_ec = (p.bits == 4u);
    threadgroup half2 lut64[64];
    if (tid < 64u) lut64[tid] = ((device const half2*)(lut_bank + (is_ec ? 1312u : 1280u)))[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half4* x_rot_h = x_rot_h_in + ulong(m) * ulong(p.padded_cols >> 2u);
    uint block_id = surv_buf[1u + surv_slot];
    uint tile = (block_id << 4u) + ((tg_pos.x & 1u) << 3u) + sg_id;
    uint r0 = tile << 2u;
    float best_v = -INFINITY;
    uint best_i = 0u;

    if (r0 < p.rows) {
        uint steps = p.padded_cols >> 8u;
        uint gh = lane >> 4u;
        device const ushort* sc0 = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw0 = packed_words + ulong(tile) * ulong(steps) * 96ul;
        float4 acc0 = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const half4* xp4 = x_rot_h + (step << 6u) + (lane << 1u);
            device const uint* p0 = pw0 + step * 96u;
            acc0 = vxt_eval_t3_lut(
                lut64, ((device const uint2*)p0)[lane], p0[64u + lane],
                *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], acc0
            );
        }
        float4 s0 = simd_sum(acc0) * 16.0f;
        if (lane == 0u) {
            for (uint k = 0u; k < 4u; ++k) {
                uint rk = r0 + k;
                if (rk < p.rows) {
                    float sc = (p.presence_pen > 0.0f) ? vxt_spec_pen_cnt(seen_vocab, rk, m, p.d1, p.d2, p.d3) : 0.0f;
                    float v = s0[k] - p.presence_pen * sc;
                    if (v > best_v || (v == best_v && rk < best_i)) { best_v = v; best_i = rk; }
                }
            }
        }
    }

    threadgroup float sg_v[8];
    threadgroup uint sg_i[8];
    vxt_stage2_reduce_tg(best_v, best_i, blk_vals, blk_idxs, sg_v, sg_i, tg_pos.x, sg_id, lane);
}

kernel void vxt_final_norm_quant_sketch(
    device float* cur_inout         [[buffer(0)]],
    device const float* ffn_out     [[buffer(1)]],
    device const float* norm_w      [[buffer(2)]],
    device float* x_rot_out         [[buffer(3)]],
    device uint4* active_grp4_out   [[buffer(4)]],
    device uint* x_qmeta_out        [[buffer(5)]],
    constant TurboNormParams& p     [[buffer(6)]],
    device half* x_rot_h_out        [[buffer(7)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]]
) {
    bool skip_add = (p.batch_m & 256u) != 0u;
    float reg_v[32];
    float sq = 0.0f;
    uint n_iters = p.padded_cols >> 8u;
    for (uint it = 0u; it < n_iters; ++it) {
        uint col = (it << 8u) + tid;
        float v = 0.0f;
        if (col < p.dim) {
            v = skip_add ? cur_inout[col] : (cur_inout[col] + ffn_out[col]);
            if (!skip_add) cur_inout[col] = v;
            sq = fma(v, v, sq);
        }
        reg_v[it] = v;
    }
    sq = simd_sum(sq);
    threadgroup float sg_tmp[8];
    if (lane == 0u) sg_tmp[sg_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float s8 = (lane < 8u) ? sg_tmp[lane] : 0.0f;
        s8 = simd_sum(s8);
        if (lane == 0u) sg_tmp[0] = rsqrt(s8 / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = sg_tmp[0];

    for (uint it = 0u; it < n_iters; ++it) {
        uint col = (it << 8u) + tid;
        float scaled = (col < p.dim) ? vxt_apply_col_sign(reg_v[it] * inv_rms * norm_w[col], col) : 0.0f;
        float rot = vxt_fwht32_lane(scaled, lane);
        x_rot_out[col] = rot;
        x_rot_h_out[col] = half(rot * 0.0625f);
    }
    threadgroup_barrier(mem_flags::mem_device);

    vxt_select_top_coords_m1(
        x_rot_out,
        active_grp4_out,
        x_qmeta_out,
        sg_tmp,
        p.padded_cols,
        tid,
        sg_id,
        lane
    );
}

kernel void vxt_final_norm_quant_wait_m1(
    VXT_COHERENT_DEV const float* cur_in [[buffer(0)]],
    device const float* norm_w           [[buffer(1)]],
    device float* x_rot_out              [[buffer(2)]],
    device uint4* active_grp4_out        [[buffer(3)]],
    device uint* x_qmeta_out             [[buffer(4)]],
    constant TurboTailNormParams& p      [[buffer(5)]],
    device half* x_rot_h_out             [[buffer(6)]],
    device atomic_uint* sync_cnt         [[buffer(7)]],
    uint tid                             [[thread_index_in_threadgroup]],
    uint sg_id                           [[simdgroup_index_in_threadgroup]],
    uint lane                            [[thread_index_in_simdgroup]]
) {
    uint n_iters = p.padded_cols >> 8u;
    float nw[16];
    for (uint it = 0u; it < n_iters; ++it) {
        uint col = (it << 8u) + tid;
        nw[it] = (col < p.rows) ? norm_w[col] : 0.0f;
    }
    if (tid == 0u) {
        while (atomic_load_explicit(&sync_cnt[p.mat_sync_slot], memory_order_relaxed) < p.next_padded_cols) {}
        atomic_store_explicit(&sync_cnt[p.mat_sync_slot], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vxt_device_fence();
    float reg_v[16];
    float sq = 0.0f;
    for (uint it = 0u; it < n_iters; ++it) {
        uint col = (it << 8u) + tid;
        float v = (col < p.rows) ? cur_in[col] : 0.0f;
        reg_v[it] = v;
        sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    threadgroup float sg_tmp[8];
    if (lane == 0u) sg_tmp[sg_id] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float s8 = (lane < 8u) ? sg_tmp[lane] : 0.0f;
        s8 = simd_sum(s8);
        if (lane == 0u) sg_tmp[0] = rsqrt(s8 / float(p.rows) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = sg_tmp[0];
    for (uint it = 0u; it < n_iters; ++it) {
        uint col = (it << 8u) + tid;
        float scaled = (col < p.rows) ? vxt_apply_col_sign(reg_v[it] * inv_rms * nw[it], col) : 0.0f;
        float rot = vxt_fwht32_lane(scaled, lane);
        x_rot_out[col] = rot;
        x_rot_h_out[col] = half(rot * 0.0625f);
    }
    threadgroup_barrier(mem_flags::mem_device);
    vxt_select_top_coords_m1(
        x_rot_out,
        active_grp4_out,
        x_qmeta_out,
        sg_tmp,
        p.padded_cols,
        tid,
        sg_id,
        lane
    );
}
