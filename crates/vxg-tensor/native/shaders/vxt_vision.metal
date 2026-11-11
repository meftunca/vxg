// ============================================================================
// vxt_vision.metal — Single-Command-Buffer Apple Metal Vision (ViT + PatchMerger)
// Supports arbitrary sequence lengths, 2D Axial RoPE, FP16 Tiled GEMM + GELU,
// and 8-SIMD-Group Online-Softmax Bidirectional Multi-Head Self-Attention.
// ============================================================================

struct VxtVisNormParams {
    uint rows;
    uint dim;
    float eps;
    uint _pad;
};

struct VxtVisGemmParams {
    uint n_rows;
    uint m_dim;
    uint k_dim;
    uint act_mode; // 0=none, 1=add_residual, 2=gelu_pytorch_tanh, 3=gelu_exact
};

struct VxtVisRopeParams {
    uint seq;
    uint nh;
    uint hd;
    uint spatial_dim;
};

struct VxtVisMhaParams {
    uint seq;
    uint nh;
    uint hd;
    float scale;
};

inline float vis_gelu_tanh(float x) {
    float z = 0.7978846f * (x + 0.044715f * x * x * x);
    float e2 = exp(-2.0f * fabs(z));
    float th = copysign((1.0f - e2) / (1.0f + e2), z);
    return 0.5f * x * (1.0f + th);
}

inline float vis_gelu_exact(float x) {
    float u = x * 0.70710678f;
    float s = (u < 0.0f) ? -1.0f : 1.0f;
    float au = fabs(u);
    float t = 1.0f / (1.0f + 0.3275911f * au);
    float poly = (((((1.0614054f * t - 1.453152f) * t) + 1.4214137f) * t - 0.28449674f) * t
        + 0.2548296f) * t;
    float erf_v = s * (1.0f - poly * exp(-au * au));
    return 0.5f * x * (1.0f + erf_v);
}

kernel void vxt_vis_add_inplace(
    device float* dst                   [[buffer(0)]],
    device const float* add_src         [[buffer(1)]],
    constant uint& count                [[buffer(2)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid < count) {
        dst[gid] += add_src[gid];
    }
}

kernel void vxt_vis_layernorm(
    device const float* x               [[buffer(0)]],
    device const float* w               [[buffer(1)]],
    device const float* b               [[buffer(2)]],
    device float* out                   [[buffer(3)]],
    constant VxtVisNormParams& p        [[buffer(4)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x;
    if (row >= p.rows) return;

    device const float* xr = x + ulong(row) * ulong(p.dim);
    device float* yr = out + ulong(row) * ulong(p.dim);
    threadgroup float sg_red[8];

    float sum = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        sum += xr[i];
    }
    sum = simd_sum(sum);
    if (lane == 0u) sg_red[sg_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float v = (lane < 8u) ? sg_red[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) sg_red[0] = v / float(p.dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = sg_red[0];

    float var = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        float d = xr[i] - mean;
        var = fma(d, d, var);
    }
    var = simd_sum(var);
    if (lane == 0u) sg_red[sg_id] = var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_id == 0u) {
        float v = (lane < 8u) ? sg_red[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) sg_red[0] = rsqrt(v / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_std = sg_red[0];

    for (uint i = tid; i < p.dim; i += 256u) {
        yr[i] = fma((xr[i] - mean) * inv_std, w[i], b[i]);
    }
}

// Batched Tiled FP16-Weight Linear: `Y[n, m] = dot(X[n, :], W_f16[m, :]) + bias[m]`
// Each SIMD group (32 lanes) computes 1 output column `m` across up to `BM = 8` token rows!
kernel void vxt_vis_gemm_f16(
    device const float* x               [[buffer(0)]],
    device const half* w_f16            [[buffer(1)]],
    device const float* bias            [[buffer(2)]],
    device float* y                     [[buffer(3)]],
    constant VxtVisGemmParams& p        [[buffer(4)]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.x * 8u + sg_id;
    uint r_base = tg_pos.y * 8u;
    if (m >= p.m_dim || r_base >= p.n_rows) return;

    uint bm = min(8u, p.n_rows - r_base);
    uint k_dim = p.k_dim;
    uint k_vec4 = k_dim & ~3u;
    device const half* wm = w_f16 + ulong(m) * ulong(k_dim);

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint k = lane * 4u; k < k_vec4; k += 128u) {
        float4 wv = float4(*(device const packed_half4*)(wm + k));
        for (uint r = 0u; r < bm; ++r) {
            device const packed_float4* xp = (device const packed_float4*)(x + ulong(r_base + r) * ulong(k_dim) + k);
            acc[r] = fma(wv.x, (*xp).x, acc[r]);
            acc[r] = fma(wv.y, (*xp).y, acc[r]);
            acc[r] = fma(wv.z, (*xp).z, acc[r]);
            acc[r] = fma(wv.w, (*xp).w, acc[r]);
        }
    }
    for (uint k = k_vec4 + lane; k < k_dim; k += 32u) {
        float wv = float(wm[k]);
        for (uint r = 0u; r < bm; ++r) {
            acc[r] = fma(wv, x[ulong(r_base + r) * ulong(k_dim) + k], acc[r]);
        }
    }

    float b_val = bias[m];
    for (uint r = 0u; r < bm; ++r) {
        float sum = simd_sum(acc[r]) + b_val;
        if (lane == 0u) {
            ulong out_idx = ulong(r_base + r) * ulong(p.m_dim) + ulong(m);
            if (p.act_mode == 1u) {
                y[out_idx] += sum;
            } else if (p.act_mode == 2u) {
                y[out_idx] = vis_gelu_tanh(sum);
            } else if (p.act_mode == 3u) {
                y[out_idx] = vis_gelu_exact(sum);
            } else {
                y[out_idx] = sum;
            }
        }
    }
}

kernel void vxt_vis_rope2d(
    device float* qkv                   [[buffer(0)]],
    device const float* rope_cos        [[buffer(1)]],
    device const float* rope_sin        [[buffer(2)]],
    constant VxtVisRopeParams& p        [[buffer(3)]],
    uint3 gid                           [[thread_position_in_grid]]
) {
    uint idx = gid.x;
    uint s = gid.y;
    uint total_per_tok = 2u * p.nh * p.spatial_dim;
    if (s >= p.seq || idx >= total_per_tok) return;

    uint qk = idx / (p.nh * p.spatial_dim);
    uint rem = idx % (p.nh * p.spatial_dim);
    uint h = rem / p.spatial_dim;
    uint i = rem % p.spatial_dim;

    uint d = p.nh * p.hd;
    device float* hv = qkv + ulong(s) * ulong(3u * d) + ulong(qk * d + h * p.hd);
    float cs = rope_cos[ulong(s) * ulong(p.spatial_dim) + i];
    float sn = rope_sin[ulong(s) * ulong(p.spatial_dim) + i];
    float x0 = hv[i];
    float x1 = hv[i + p.spatial_dim];
    hv[i] = x0 * cs - x1 * sn;
    hv[i + p.spatial_dim] = x1 * cs + x0 * sn;
}

// 8-SIMD-group online-softmax bidirectional self-attention (`grid = (nh, seq, 1)`, `tg = 256`)
kernel void vxt_vis_mha(
    device const float* qkv             [[buffer(0)]],
    device float* attn_cat              [[buffer(1)]],
    constant VxtVisMhaParams& p         [[buffer(2)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    uint qi = tg_pos.y;
    if (h >= p.nh || qi >= p.seq) return;

    uint hd = p.hd;
    uint d = p.nh * hd;
    ulong row_stride = ulong(3u * d);
    uint q_off = h * hd;
    uint k_off = d + h * hd;
    uint v_off = 2u * d + h * hd;

    threadgroup float q_sh[256];
    threadgroup float sg_m[8];
    threadgroup float sg_l[8];
    threadgroup float sg_acc[8 * 256];

    device const float* q_ptr = qkv + ulong(qi) * row_stride + q_off;
    for (uint j = tid; j < hd; j += 256u) {
        q_sh[j] = q_ptr[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint hd_chunks = (hd + 31u) >> 5u;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint ki = sg_id; ki < p.seq; ki += 8u) {
        device const float* kh = qkv + ulong(ki) * row_stride + k_off;
        device const float* vh = qkv + ulong(ki) * row_stride + v_off;
        float dot = 0.0f;
        for (uint c = 0u; c < hd_chunks; ++c) {
            uint j = (c << 5u) + lane;
            if (j < hd) dot = fma(q_sh[j], kh[j], dot);
        }
        float s = simd_sum(dot) * p.scale;
        float m_new = max(m_i, s);
        float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
        float w = exp(s - m_new);
        l_i = fma(l_i, alpha, w);
        m_i = m_new;
        for (uint c = 0u; c < hd_chunks; ++c) {
            uint j = (c << 5u) + lane;
            if (j < hd) acc[c] = fma(w, vh[j], acc[c] * alpha);
        }
    }

    if (lane == 0u) {
        sg_m[sg_id] = m_i;
        sg_l[sg_id] = l_i;
    }
    for (uint c = 0u; c < hd_chunks; ++c) {
        uint j = (c << 5u) + lane;
        if (j < hd) sg_acc[sg_id * 256u + j] = acc[c];
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

    device float* dst = attn_cat + ulong(qi) * ulong(d) + q_off;
    for (uint j = tid; j < hd; j += 256u) {
        float out_j = 0.0f;
        for (uint sg = 0u; sg < 8u; ++sg) {
            out_j = fma(sg_acc[sg * 256u + j], w_sg[sg], out_j);
        }
        dst[j] = out_j * inv_l;
    }
}
