// ============================================================================
// vxt_cb4_fast_ops.metal — Fused Exact Embedding, Add+RMSNorm, 64-TG Split-State
// FP32 DeltaNet, and 2-Stage Parallel Argmax (0-ULP Bit-Identical)
// ============================================================================

struct Cb4FastEmbedParams {
    uint cols, padded_cols, batch_m, n_vocab;
    float eps, _pad0, _pad1, _pad2;
    float centroids[16];
};
struct Cb4FastNormParams { uint dim, padded_dim, batch_m; float eps; };
struct Cb4FastSwigluParams { uint n_ff, gu_stride, padded_ff, batch_m; };
struct Cb4FastAbParams {
    uint rows, padded_cols, x_stride, y_stride, ab_half_stride, n_vh, batch_m, is_strided32;
};
struct Cb4FastDnetM1Params {
    uint n_kh, n_vh, dk, dv, conv_ch, k_sz, conv_tgs, padded_cols, ab_half_stride, padded_inner, is_strided32;
    float eps;
};
struct Cb4FastArgmaxParams { uint n_vocab, slot, next_slot, num_tgs; };

inline float cb4_decode_embed_elem(
    device const ushort* tiled_scales,
    device const uint* tiled_words,
    device const uint* row_ptr,
    device const uint* outliers,
    constant const float* centroids,
    uint tok,
    uint col,
    uint padded_cols
) {
    uint groups_per_row = padded_cols >> 7u;
    uint steps = groups_per_row >> 1u;
    uint tile = tok >> 2u;
    uint r_sub = tok & 3u;
    uint grp = col >> 7u;
    uint step = col >> 8u;
    uint lane = (col & 255u) >> 3u;
    uint nib = col & 7u;

    ushort sc_u16 = tiled_scales[((ulong(tile) * ulong(groups_per_row) + ulong(grp)) << 2u) + r_sub];
    uint word = tiled_words[(((ulong(tile) * ulong(steps) + ulong(step)) * 32ul + ulong(lane)) << 2u) + r_sub];
    uint q = (word >> (nib << 2u)) & 15u;
    volatile float value = float(as_type<half>(sc_u16)) * centroids[q];
    for (uint k = row_ptr[tok]; k < row_ptr[tok + 1u]; ++k) {
        uint packed = outliers[k];
        if ((packed & 65535u) == col) {
            value = value + float(as_type<half>(ushort(packed >> 16u)));
        }
    }
    return value;
}

inline float cb4_ab_dot_exact(
    device const float* w_row,
    device const float* xm,
    uint padded_cols,
    uint is_strided32,
    uint lane
) {
    if (is_strided32 != 0u) {
        float acc = 0.0f;
        for (uint c = lane; c < padded_cols; c += 32u) {
            acc = fma(w_row[c], xm[c], acc);
        }
        return simd_sum(acc);
    }
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint sbs = padded_cols >> 8u;
    for (uint sb = 0u; sb < sbs; ++sb) {
        uint off = (sb << 8u) + (lane << 3u);
        device const float4* wp4 = (device const float4*)(w_row + off);
        device const float4* xp4 = (device const float4*)(xm + off);
        float4 w0 = wp4[0], w1 = wp4[1];
        float4 x0 = xp4[0], x1 = xp4[1];
        acc0 = fma(w0.x, x0.x, acc0);
        acc1 = fma(w0.y, x0.y, acc1);
        acc0 = fma(w0.z, x0.z, acc0);
        acc1 = fma(w0.w, x0.w, acc1);
        acc0 = fma(w1.x, x1.x, acc0);
        acc1 = fma(w1.y, x1.y, acc1);
        acc0 = fma(w1.z, x1.z, acc0);
        acc1 = fma(w1.w, x1.w, acc1);
    }
    return simd_sum(acc0 + acc1);
}

kernel void vxt_cb4_embed_exact_batch(
    device const ushort* tiled_scales   [[buffer(0)]],
    device const uint* tiled_words      [[buffer(1)]],
    device const uint* row_ptr          [[buffer(2)]],
    device const uint* outliers         [[buffer(3)]],
    device const uint* tok_buf          [[buffer(4)]],
    device float* cur_out               [[buffer(5)]],
    constant Cb4FastEmbedParams& p      [[buffer(6)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint col = tg_pos.x * 256u + tid;
    uint m = tg_pos.y;
    if (m >= p.batch_m || col >= p.padded_cols) return;
    device float* dst = cur_out + ulong(m) * ulong(p.padded_cols);
    if (col >= p.cols) {
        dst[col] = 0.0f;
        return;
    }
    uint tok = min(tok_buf[m], p.n_vocab - 1u);
    dst[col] = cb4_decode_embed_elem(
        tiled_scales, tiled_words, row_ptr, outliers, p.centroids, tok, col, p.padded_cols
    );
}

// Single-dispatch M=1 Embedding Decode + Layer-0 RMSNorm (1 TG x 256 threads, 0-ULP exact).
kernel void vxt_cb4_embed_norm0_m1(
    device const ushort* tiled_scales   [[buffer(0)]],
    device const uint* tiled_words      [[buffer(1)]],
    device const uint* row_ptr          [[buffer(2)]],
    device const uint* outliers         [[buffer(3)]],
    device const uint* tok_buf          [[buffer(4)]],
    device const float* norm0_w         [[buffer(5)]],
    device float* cur_out               [[buffer(6)]],
    device float* norm_out              [[buffer(7)]],
    constant Cb4FastEmbedParams& p      [[buffer(8)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]]
) {
    uint tok = min(tok_buf[0], p.n_vocab - 1u);
    float sq = 0.0f;
    for (uint col = tid; col < p.padded_cols; col += 256u) {
        if (col < p.cols) {
            float v = cb4_decode_embed_elem(
                tiled_scales, tiled_words, row_ptr, outliers, p.centroids, tok, col, p.padded_cols
            );
            cur_out[col] = v;
            sq = fma(v, v, sq);
        } else {
            cur_out[col] = 0.0f;
        }
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
    float scale = sg_sums[0];
    for (uint col = tid; col < p.padded_cols; col += 256u) {
        norm_out[col] = (col < p.cols) ? (cur_out[col] * scale * norm0_w[col]) : 0.0f;
    }
}

kernel void vxt_cb4_rms_norm_batch(
    device const float* x               [[buffer(0)]],
    device const float* w               [[buffer(1)]],
    device float* out                   [[buffer(2)]],
    constant Cb4FastNormParams& p       [[buffer(3)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    if (m >= p.batch_m) return;
    device const float* xm = x + ulong(m) * ulong(p.padded_dim);
    device float* ym = out + ulong(m) * ulong(p.padded_dim);

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
    for (uint i = tid; i < p.padded_dim; i += 256u) {
        ym[i] = (i < p.dim) ? (xm[i] * scale * w[i]) : 0.0f;
    }
}

// Fuses residual `cur += delta` with next `RMSNorm(cur)` — eliminates cross-core RMW contention on `cur`!
kernel void vxt_cb4_add_rms_norm_batch(
    device float* cur                   [[buffer(0)]],
    device const float* delta           [[buffer(1)]],
    device const float* w               [[buffer(2)]],
    device float* out                   [[buffer(3)]],
    constant Cb4FastNormParams& p       [[buffer(4)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint m = tg_pos.y;
    if (m >= p.batch_m) return;
    device float* cm = cur + ulong(m) * ulong(p.padded_dim);
    device const float* dm = delta + ulong(m) * ulong(p.padded_dim);
    device float* ym = out + ulong(m) * ulong(p.padded_dim);

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
    for (uint i = tid; i < p.padded_dim; i += 256u) {
        ym[i] = (i < p.dim) ? (cm[i] * scale * w[i]) : 0.0f;
    }
}

kernel void vxt_cb4_swiglu_batch(
    device const float* gu_in           [[buffer(0)]],
    device float* act_out               [[buffer(1)]],
    constant Cb4FastSwigluParams& p     [[buffer(2)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint col = tg_pos.x * 256u + tid;
    uint m = tg_pos.y;
    if (m >= p.batch_m || col >= p.padded_ff) return;
    device float* dst = act_out + ulong(m) * ulong(p.padded_ff);
    if (col >= p.n_ff) {
        dst[col] = 0.0f;
        return;
    }
    device const float* src = gu_in + ulong(m) * ulong(p.gu_stride);
    float g = src[col];
    float u = src[p.n_ff + col];
    dst[col] = (g / (1.0f + exp(-g))) * u;
}

kernel void vxt_cb4_ab_gemv_batch(
    device const float* weights         [[buffer(0)]],
    device const float* x_in            [[buffer(1)]],
    device float* ab_out                [[buffer(2)]],
    constant Cb4FastAbParams& p         [[buffer(3)]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 8u + sg_id;
    uint m = tg_pos.y;
    if (m >= p.batch_m || row >= p.rows) return;

    device const float* w_row = weights + ulong(row) * ulong(p.padded_cols);
    device const float* xm = x_in + ulong(m) * ulong(p.x_stride);
    float sum = cb4_ab_dot_exact(w_row, xm, p.padded_cols, p.is_strided32, lane);

    if (lane == 0u) {
        device float* ym = ab_out + ulong(m) * ulong(p.y_stride);
        if (row < p.n_vh) {
            ym[row] = sum;
        } else {
            ym[p.ab_half_stride + (row - p.n_vh)] = sum;
        }
    }
}

// Single-wave M=1 Fused DeltaNet Conv1D (TGs 0..conv_tgs-1) + [ssm_alpha; ssm_beta] GEMV (TGs conv_tgs..).
kernel void vxt_cb4_dnet_prep_m1(
    device const float* qkv_z           [[buffer(0)]],
    device const float* ssm_conv1d      [[buffer(1)]],
    device float* conv_state            [[buffer(2)]],
    device float* conv_silu             [[buffer(3)]],
    device const float* ab_weights      [[buffer(4)]],
    device const float* x_norm          [[buffer(5)]],
    device float* ab_out                [[buffer(6)]],
    constant Cb4FastDnetM1Params& p     [[buffer(7)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    if (tg_pos.x < p.conv_tgs) {
        uint c = tg_pos.x * 256u + tid;
        if (c >= p.conv_ch) return;
        uint k_sz = p.k_sz;
        uint hist = k_sz - 1u;
        device const float* w = ssm_conv1d + c * k_sz;
        float x_c = qkv_z[c];
        float acc = 0.0f;
        for (uint t = 0u; t < hist; ++t) {
            acc = fma(conv_state[t * p.conv_ch + c], w[t], acc);
        }
        acc = fma(x_c, w[hist], acc);
        if (hist > 0u) {
            for (uint t = 0u; t + 1u < hist; ++t) {
                conv_state[t * p.conv_ch + c] = conv_state[(t + 1u) * p.conv_ch + c];
            }
            conv_state[(hist - 1u) * p.conv_ch + c] = x_c;
        }
        conv_silu[c] = acc / (1.0f + exp(-acc));
    } else {
        uint row = (tg_pos.x - p.conv_tgs) * 8u + sg_id;
        if (row >= 2u * p.n_vh) return;
        device const float* w_row = ab_weights + ulong(row) * ulong(p.padded_cols);
        float sum = cb4_ab_dot_exact(w_row, x_norm, p.padded_cols, p.is_strided32, lane);
        if (lane == 0u) {
            if (row < p.n_vh) ab_out[row] = sum;
            else ab_out[p.ab_half_stride + (row - p.n_vh)] = sum;
        }
    }
}

// 64-TG (n_vh * 4) Split-State M=1 FP32 DeltaNet State Update — saturates all GPU cores with 0-ULP identity!
kernel void vxt_cb4_dnet_state_split_m1(
    device const float* conv_silu       [[buffer(0)]],
    device const float* ab_in           [[buffer(1)]],
    device const float* ssm_a           [[buffer(2)]],
    device const float* ssm_dt_bias     [[buffer(3)]],
    device float* ssm_state             [[buffer(4)]],
    device float* o_raw                 [[buffer(5)]],
    constant Cb4FastDnetM1Params& p     [[buffer(6)]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x >> 2u;
    uint sp_idx = tg_pos.x & 3u;
    if (h >= p.n_vh) return;

    uint dk = p.dk;
    uint dv = p.dv;
    uint rows_per_split = dv >> 2u;
    uint jv_base = sp_idx * rows_per_split;
    uint hk = h / (p.n_vh / p.n_kh);
    uint qk_dim = p.n_kh * dk;

    device const float* q_src = conv_silu + hk * dk;
    device const float* k_src = conv_silu + qk_dim + hk * dk;
    device const float* v_h = conv_silu + 2u * qk_dim + h * dv;

    threadgroup float q_sh[256];
    threadgroup float k_sh[256];
    if (sg_id == 0u) {
        float sq = 0.0f;
        for (uint i = lane; i < dk; i += 32u) { float v = q_src[i]; sq = fma(v, v, sq); }
        float sc = rsqrt(float(dk)) * rsqrt(simd_sum(sq) + p.eps);
        for (uint i = lane; i < dk; i += 32u) q_sh[i] = q_src[i] * sc;
    } else if (sg_id == 1u) {
        float sq = 0.0f;
        for (uint i = lane; i < dk; i += 32u) { float v = k_src[i]; sq = fma(v, v, sq); }
        float sc = rsqrt(simd_sum(sq) + p.eps);
        for (uint i = lane; i < dk; i += 32u) k_sh[i] = k_src[i] * sc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float beta = 1.0f / (1.0f + exp(-ab_in[p.ab_half_stride + h]));
    float x_sp = ab_in[h] + ssm_dt_bias[h];
    float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? exp(x_sp) : log(1.0f + exp(x_sp)));
    float decay = exp(sp * ssm_a[h]);

    device float* sh_base = ssm_state + ulong(h) * ulong(dv) * ulong(dk);
    if (dk == 128u) {
        uint i0 = lane, i1 = lane + 32u, i2 = lane + 64u, i3 = lane + 96u;
        float k0 = k_sh[i0], k1 = k_sh[i1], k2 = k_sh[i2], k3 = k_sh[i3];
        float q0 = q_sh[i0], q1 = q_sh[i1], q2 = q_sh[i2], q3 = q_sh[i3];
        for (uint r = sg_id; r < rows_per_split; r += 4u) {
            uint jv = jv_base + r;
            device float* row = sh_base + (jv << 7u);
            float r0 = row[i0] * decay, r1 = row[i1] * decay;
            float r2 = row[i2] * decay, r3 = row[i3] * decay;
            float sk = simd_sum(fma(r3, k3, fma(r2, k2, fma(r1, k1, r0 * k0))));
            float d_jv = (v_h[jv] - sk) * beta;
            r0 = fma(d_jv, k0, r0); r1 = fma(d_jv, k1, r1);
            r2 = fma(d_jv, k2, r2); r3 = fma(d_jv, k3, r3);
            row[i0] = r0; row[i1] = r1; row[i2] = r2; row[i3] = r3;
            float o_jv = simd_sum(fma(r3, q3, fma(r2, q2, fma(r1, q1, r0 * q0))));
            if (lane == 0u) o_raw[h * dv + jv] = o_jv;
        }
    } else {
        uint dk_chunks = (dk + 31u) >> 5u;
        for (uint r = sg_id; r < rows_per_split; r += 4u) {
            uint jv = jv_base + r;
            device float* row = sh_base + ulong(jv) * ulong(dk);
            float r_reg[8];
            float sk = 0.0f;
            for (uint c = 0u; c < dk_chunks; ++c) {
                uint ik = (c << 5u) + lane;
                if (ik < dk) {
                    float sd = row[ik] * decay;
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
                    row[ik] = sn;
                    o_jv = fma(sn, q_sh[ik], o_jv);
                }
            }
            o_jv = simd_sum(o_jv);
            if (lane == 0u) o_raw[h * dv + jv] = o_jv;
        }
    }
}

// Exact 8-SG RMSNorm + SiLU(z) epilogue for Split-State DeltaNet (0-ULP to vxt_deltanet_heads).
kernel void vxt_cb4_dnet_out_m1(
    device const float* o_raw           [[buffer(0)]],
    device const float* qkv_z           [[buffer(1)]],
    device const float* ssm_norm        [[buffer(2)]],
    device float* y_concat              [[buffer(3)]],
    constant Cb4FastDnetM1Params& p     [[buffer(4)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    if (h >= p.n_vh) return;
    uint dv = p.dv;
    threadgroup float sg_red[8];

    float sq_o = 0.0f;
    for (uint jv = tid; jv < dv; jv += 256u) {
        float v = o_raw[h * dv + jv];
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
    device const float* z_h = qkv_z + p.conv_ch + h * dv;
    device float* dst_h = y_concat + h * dv;
    for (uint jv = tid; jv < dv; jv += 256u) {
        float zv = z_h[jv];
        dst_h[jv] = (o_raw[h * dv + jv] * o_scale * ssm_norm[jv]) * (zv / (1.0f + exp(-zv)));
    }
    if (h == 0u && p.n_vh * dv < p.padded_inner) {
        for (uint col = p.n_vh * dv + tid; col < p.padded_inner; col += 256u) {
            y_concat[col] = 0.0f;
        }
    }
}

kernel void vxt_cb4_argmax_s1(
    device const float* logits          [[buffer(0)]],
    device float* s1_vals               [[buffer(1)]],
    device uint* s1_ids                 [[buffer(2)]],
    device uint* s1_bad                 [[buffer(3)]],
    constant Cb4FastArgmaxParams& p     [[buffer(4)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    threadgroup float values[256];
    threadgroup uint ids[256];
    threadgroup uint invalid[256];

    uint tg_id = tg_pos.x;
    uint stride_all = p.num_tgs * 256u;
    uint nonfinite = 0u;
    float best = -INFINITY;
    uint best_id = 0xffffffffu;

    for (uint i = tg_id * 256u + tid; i < p.n_vocab; i += stride_all) {
        float v = logits[i];
        nonfinite |= uint(!isfinite(v));
        if (v > best || (v == best && i < best_id)) {
            best = v;
            best_id = i;
        }
    }

    values[tid] = best;
    ids[tid] = best_id;
    invalid[tid] = nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            invalid[tid] |= invalid[tid + stride];
            float v = values[tid + stride];
            uint i = ids[tid + stride];
            if (v > values[tid] || (v == values[tid] && i < ids[tid])) {
                values[tid] = v;
                ids[tid] = i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        s1_vals[tg_id] = values[0];
        s1_ids[tg_id] = ids[0];
        s1_bad[tg_id] = invalid[0];
    }
}

kernel void vxt_cb4_argmax_s2(
    device const float* s1_vals         [[buffer(0)]],
    device const uint* s1_ids           [[buffer(1)]],
    device const uint* s1_bad           [[buffer(2)]],
    device uint* out_ring               [[buffer(3)]],
    device uint* err_ring               [[buffer(4)]],
    device uint* tok_ring               [[buffer(5)]],
    constant Cb4FastArgmaxParams& p     [[buffer(6)]],
    uint tid                            [[thread_index_in_threadgroup]]
) {
    threadgroup float values[256];
    threadgroup uint ids[256];
    threadgroup uint invalid[256];

    if (tid < p.num_tgs) {
        values[tid] = s1_vals[tid];
        ids[tid] = s1_ids[tid];
        invalid[tid] = s1_bad[tid];
    } else {
        values[tid] = -INFINITY;
        ids[tid] = 0xffffffffu;
        invalid[tid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            invalid[tid] |= invalid[tid + stride];
            float v = values[tid + stride];
            uint i = ids[tid + stride];
            if (v > values[tid] || (v == values[tid] && i < ids[tid])) {
                values[tid] = v;
                ids[tid] = i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        uint winner = (ids[0] < p.n_vocab) ? ids[0] : 0u;
        out_ring[p.slot] = winner;
        err_ring[p.slot] = invalid[0];
        if (p.next_slot != 0xffffffffu) {
            tok_ring[p.next_slot] = winner;
        }
    }
}
