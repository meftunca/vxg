// ============================================================================
// vxt_llm_ops.metal — All-GPU 1-Command-Buffer LLM Operators
// (`RMSNorm`, `Add+RMSNorm`, `SwiGLU`, `Gated-DeltaNet`, `Flash-Decoding GQA`)
// ============================================================================

struct RmsNormParams {
    uint dim;
    float eps;
};

struct DeltaConvParams {
    uint conv_ch;
    uint k_sz;
};

struct DeltaHeadsParams {
    uint n_kh;
    uint n_vh;
    uint dk;
    uint dv;
    float eps;
};

struct GqaParams {
    uint nh;
    uint nkv;
    uint hd;
    uint pos;
    uint rope_dim;
    uint flags; // bit0=gated_q, bit1=rope_neox, bit2=q_norm, bit3=k_norm, bit4=q_bias, bit5=k_bias, bit6=v_bias
    float rope_freq_base;
    float eps;
};

kernel void vxt_gemv_raw32(
    device const float* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float* y             [[buffer(2)]],
    constant Q4_0Params& p      [[buffer(3)]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint3 tg_pos                [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 2u + sg_id;
    if (row >= p.rows) return;
    device const float* w_row = weights + ulong(row) * ulong(p.padded_cols);
    float acc = 0.0f;
    for (uint c = lane; c < p.padded_cols; c += 32u) {
        acc = fma(w_row[c], x[c], acc);
    }
    float sum = simd_sum(acc);
    if (lane == 0u) {
        y[row] = sum;
    }
}

kernel void vxt_rms_norm(
    device const float* x       [[buffer(0)]],
    device const float* w       [[buffer(1)]],
    device float* out           [[buffer(2)]],
    constant RmsNormParams& p   [[buffer(3)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]]
) {
    float sq = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        float v = x[i];
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
    for (uint i = tid; i < p.dim; i += 256u) {
        out[i] = x[i] * scale * w[i];
    }
}

kernel void vxt_add_rms_norm(
    device float* cur           [[buffer(0)]],
    device const float* delta   [[buffer(1)]],
    device const float* w       [[buffer(2)]],
    device float* out           [[buffer(3)]],
    constant RmsNormParams& p   [[buffer(4)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]]
) {
    float sq = 0.0f;
    for (uint i = tid; i < p.dim; i += 256u) {
        float v = cur[i] + delta[i];
        cur[i] = v;
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
    for (uint i = tid; i < p.dim; i += 256u) {
        out[i] = cur[i] * scale * w[i];
    }
}

kernel void vxt_vec_add(
    device float* cur           [[buffer(0)]],
    device const float* delta   [[buffer(1)]],
    constant uint& dim          [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid < dim) {
        cur[tid] += delta[tid];
    }
}

kernel void vxt_swiglu_mul(
    device float* gate          [[buffer(0)]],
    device const float* up      [[buffer(1)]],
    constant uint& n_ff         [[buffer(2)]],
    uint tid                    [[thread_position_in_grid]]
) {
    if (tid < n_ff) {
        float g = gate[tid];
        gate[tid] = (g / (1.0f + exp(-g))) * up[tid];
    }
}

kernel void vxt_deltanet_conv(
    device const float* qkv_mixed   [[buffer(0)]],
    device const float* ssm_conv1d  [[buffer(1)]],
    device float* conv_state        [[buffer(2)]],
    device float* conv_silu         [[buffer(3)]],
    constant DeltaConvParams& p     [[buffer(4)]],
    uint c                          [[thread_position_in_grid]]
) {
    if (c >= p.conv_ch) return;
    uint k_sz = p.k_sz;
    uint hist = k_sz - 1u;
    device const float* w = ssm_conv1d + c * k_sz;
    float x_c = qkv_mixed[c];
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
}

kernel void vxt_deltanet_heads(
    device const float* conv_silu   [[buffer(0)]],
    device const float* z           [[buffer(1)]],
    device const float* alpha_raw   [[buffer(2)]],
    device const float* beta_raw    [[buffer(3)]],
    device const float* ssm_a       [[buffer(4)]],
    device const float* ssm_dt_bias [[buffer(5)]],
    device const float* ssm_norm    [[buffer(6)]],
    device float* ssm_state         [[buffer(7)]],
    device float* y_concat          [[buffer(8)]],
    constant DeltaHeadsParams& p    [[buffer(9)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    if (h >= p.n_vh) return;

    uint dk = p.dk;
    uint dv = p.dv;
    uint hk = h / (p.n_vh / p.n_kh);
    uint qk_dim = p.n_kh * dk;

    device const float* q_src = conv_silu + hk * dk;
    device const float* k_src = conv_silu + qk_dim + hk * dk;
    device const float* v_h = conv_silu + 2u * qk_dim + h * dv;

    threadgroup float q_sh[512];
    threadgroup float k_sh[512];
    threadgroup float o_sh[512];
    threadgroup float sg_red[8];

    if (sg_id == 0u) {
        float sq = 0.0f;
        for (uint i = lane; i < dk; i += 32u) {
            float v = q_src[i];
            sq = fma(v, v, sq);
        }
        sq = simd_sum(sq);
        float sc = rsqrt(float(dk)) * rsqrt(sq + p.eps);
        for (uint i = lane; i < dk; i += 32u) {
            q_sh[i] = q_src[i] * sc;
        }
    } else if (sg_id == 1u) {
        float sq = 0.0f;
        for (uint i = lane; i < dk; i += 32u) {
            float v = k_src[i];
            sq = fma(v, v, sq);
        }
        sq = simd_sum(sq);
        float sc = rsqrt(sq + p.eps);
        for (uint i = lane; i < dk; i += 32u) {
            k_sh[i] = k_src[i] * sc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float beta = 1.0f / (1.0f + exp(-beta_raw[h]));
    float x_sp = alpha_raw[h] + ssm_dt_bias[h];
    float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? exp(x_sp) : log(1.0f + exp(x_sp)));
    float decay = exp(sp * ssm_a[h]);

    device float* sh_base = ssm_state + ulong(h) * ulong(dv) * ulong(dk);
    if (dk == 128u) {
        uint i0 = lane, i1 = lane + 32u, i2 = lane + 64u, i3 = lane + 96u;
        float k0 = k_sh[i0], k1 = k_sh[i1], k2 = k_sh[i2], k3 = k_sh[i3];
        float q0 = q_sh[i0], q1 = q_sh[i1], q2 = q_sh[i2], q3 = q_sh[i3];
        for (uint jv = sg_id; jv < dv; jv += 8u) {
            device float* row = sh_base + (jv << 7u);
            float r0 = row[i0] * decay;
            float r1 = row[i1] * decay;
            float r2 = row[i2] * decay;
            float r3 = row[i3] * decay;
            float sk = simd_sum(fma(r3, k3, fma(r2, k2, fma(r1, k1, r0 * k0))));
            float d_jv = (v_h[jv] - sk) * beta;
            r0 = fma(d_jv, k0, r0);
            r1 = fma(d_jv, k1, r1);
            r2 = fma(d_jv, k2, r2);
            r3 = fma(d_jv, k3, r3);
            row[i0] = r0;
            row[i1] = r1;
            row[i2] = r2;
            row[i3] = r3;
            float o_jv = simd_sum(fma(r3, q3, fma(r2, q2, fma(r1, q1, r0 * q0))));
            if (lane == 0u) o_sh[jv] = o_jv;
        }
    } else {
        uint dk_chunks = (dk + 31u) >> 5u;
        for (uint jv = sg_id; jv < dv; jv += 8u) {
            device float* row = sh_base + ulong(jv) * ulong(dk);
            float r_reg[16];
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
    device const float* z_h = z + h * dv;
    device float* dst_h = y_concat + h * dv;
    for (uint jv = tid; jv < dv; jv += 256u) {
        float zv = z_h[jv];
        dst_h[jv] = (o_sh[jv] * o_scale * ssm_norm[jv]) * (zv / (1.0f + exp(-zv)));
    }
}

kernel void vxt_gqa_prep(
    device const float* q_raw   [[buffer(0)]],
    device const float* k_cur   [[buffer(1)]],
    device const float* v_cur   [[buffer(2)]],
    device const float* q_norm  [[buffer(3)]],
    device const float* k_norm  [[buffer(4)]],
    device const float* q_bias  [[buffer(5)]],
    device const float* k_bias  [[buffer(6)]],
    device const float* v_bias  [[buffer(7)]],
    device float* q_cur_out     [[buffer(8)]],
    device float* q_gate_out    [[buffer(9)]],
    device float* k_cache       [[buffer(10)]],
    device float* v_cache       [[buffer(11)]],
    constant GqaParams& p       [[buffer(12)]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint3 tg_pos                [[threadgroup_position_in_grid]]
) {
    uint idx = tg_pos.x;
    uint hd = p.hd;
    bool gated_q = (p.flags & 1u) != 0u;
    bool neox = (p.flags & 2u) != 0u;
    bool has_qn = (p.flags & 4u) != 0u;
    bool has_kn = (p.flags & 8u) != 0u;
    bool has_qb = (p.flags & 16u) != 0u;
    bool has_kb = (p.flags & 32u) != 0u;
    bool has_vb = (p.flags & 64u) != 0u;

    threadgroup float h_buf[512];

    if (idx < p.nh) {
        uint h = idx;
        if (gated_q) {
            uint base = h * 2u * hd;
            for (uint d = lane; d < hd; d += 32u) {
                float qb0 = has_qb ? q_bias[base + d] : 0.0f;
                float qb1 = has_qb ? q_bias[base + hd + d] : 0.0f;
                h_buf[d] = q_raw[base + d] + qb0;
                q_gate_out[h * hd + d] = q_raw[base + hd + d] + qb1;
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
            for (uint d = lane; d < hd; d += 32u) {
                h_buf[d] *= sc * q_norm[d];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint rdim = min(p.rope_dim, hd);
        uint half_r = rdim >> 1u;
        for (uint i = lane; i < half_r; i += 32u) {
            float freq = pow(p.rope_freq_base, -2.0f * float(i) / float(rdim));
            float theta = float(p.pos) * freq;
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
            q_cur_out[h * hd + d] = h_buf[d];
        }
    } else if (idx < p.nh + p.nkv) {
        uint kh = idx - p.nh;
        uint base = kh * hd;
        ulong cache_off = ulong(p.pos) * ulong(p.nkv * hd) + ulong(base);
        for (uint d = lane; d < hd; d += 32u) {
            h_buf[d] = k_cur[base + d] + (has_kb ? k_bias[base + d] : 0.0f);
            v_cache[cache_off + d] = v_cur[base + d] + (has_vb ? v_bias[base + d] : 0.0f);
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
            for (uint d = lane; d < hd; d += 32u) {
                h_buf[d] *= sc * k_norm[d];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint rdim = min(p.rope_dim, hd);
        uint half_r = rdim >> 1u;
        for (uint i = lane; i < half_r; i += 32u) {
            float freq = pow(p.rope_freq_base, -2.0f * float(i) / float(rdim));
            float theta = float(p.pos) * freq;
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
            k_cache[cache_off + d] = h_buf[d];
        }
    }
}

kernel void vxt_gqa_attn(
    device const float* q_cur   [[buffer(0)]],
    device const float* q_gate  [[buffer(1)]],
    device const float* k_cache [[buffer(2)]],
    device const float* v_cache [[buffer(3)]],
    device float* att_cat       [[buffer(4)]],
    constant GqaParams& p       [[buffer(5)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint sg_id                  [[simdgroup_index_in_threadgroup]],
    uint lane                   [[thread_index_in_simdgroup]],
    uint3 tg_pos                [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    if (h >= p.nh) return;

    uint hd = p.hd;
    uint kv_h = h / (p.nh / p.nkv);
    uint seq_len = p.pos + 1u;
    ulong kv_stride = ulong(p.nkv * hd);
    float scale = rsqrt(float(hd));
    bool gated_q = (p.flags & 1u) != 0u;

    threadgroup float q_sh[512];
    threadgroup float sg_m[8];
    threadgroup float sg_l[8];
    threadgroup float sg_acc[8 * 512];

    for (uint d = tid; d < hd; d += 256u) {
        q_sh[d] = q_cur[h * hd + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint hd_chunks = (hd + 31u) >> 5u;
    float acc[16];
    for (uint c = 0u; c < hd_chunks; ++c) {
        acc[c] = 0.0f;
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint t = sg_id; t < seq_len; t += 8u) {
        ulong base = ulong(t) * kv_stride + ulong(kv_h * hd);
        device const float* kh = k_cache + base;
        device const float* vh = v_cache + base;
        float dot = 0.0f;
        for (uint c = 0u; c < hd_chunks; ++c) {
            uint d = (c << 5u) + lane;
            if (d < hd) {
                dot = fma(q_sh[d], kh[d], dot);
            }
        }
        float s = simd_sum(dot) * scale;
        float m_new = max(m_i, s);
        float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
        float w = exp(s - m_new);
        l_i = fma(l_i, alpha, w);
        m_i = m_new;
        for (uint c = 0u; c < hd_chunks; ++c) {
            uint d = (c << 5u) + lane;
            if (d < hd) {
                acc[c] = fma(w, vh[d], acc[c] * alpha);
            }
        }
    }

    if (lane == 0u) {
        sg_m[sg_id] = m_i;
        sg_l[sg_id] = l_i;
    }
    for (uint c = 0u; c < hd_chunks; ++c) {
        uint d = (c << 5u) + lane;
        if (d < hd) {
            sg_acc[sg_id * 512u + d] = acc[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_all = -INFINITY;
    for (uint sg = 0u; sg < 8u; ++sg) {
        if (sg_l[sg] > 0.0f) {
            m_all = max(m_all, sg_m[sg]);
        }
    }
    float w_sg[8];
    float l_all = 0.0f;
    for (uint sg = 0u; sg < 8u; ++sg) {
        float w = (sg_l[sg] > 0.0f) ? exp(sg_m[sg] - m_all) : 0.0f;
        w_sg[sg] = w;
        l_all = fma(sg_l[sg], w, l_all);
    }
    float inv_l = 1.0f / l_all;

    for (uint d = tid; d < hd; d += 256u) {
        float out_d = 0.0f;
        for (uint sg = 0u; sg < 8u; ++sg) {
            out_d = fma(sg_acc[sg * 512u + d], w_sg[sg], out_d);
        }
        out_d *= inv_l;
        if (gated_q) {
            float g = q_gate[h * hd + d];
            out_d *= 1.0f / (1.0f + exp(-g));
        }
        att_cat[h * hd + d] = out_d;
    }
}

// Decode the same learned centroid table and additive CSR outliers as CPU
// CodebookMatrix::decode_row. This is deliberately independent of the polynomial
// approximation used by GEMV. Volatile forces the CPU's multiply-then-add rounding.
kernel void vxt_embedding_cb4_exact(
    device const ushort* scales [[buffer(0)]],
    device const uint* words [[buffer(1)]],
    device const uint* row_ptr [[buffer(2)]],
    device const uint* outliers [[buffer(3)]],
    device float* dst [[buffer(4)]],
    constant uint* p [[buffer(5)]], // token, logical columns, padded columns
    constant float* centroids [[buffer(6)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= p[1]) return;
    ulong group = ulong(p[0]) * (p[2] / 128u) + col / 128u;
    uint word = words[group * 16ul + (col % 128u) / 8u];
    uint q = (word >> ((col % 8u) * 4u)) & 15u;
    volatile float value = float(as_type<half>(scales[group])) * centroids[q];
    for (uint k = row_ptr[p[0]]; k < row_ptr[p[0] + 1u]; ++k) {
        uint packed = outliers[k];
        if ((packed & 65535u) == col)
            value = value + float(as_type<half>(ushort(packed >> 16u)));
    }
    dst[col] = value;
}

// Strict > with lowest-index tie matches the ordinary greedy sampler for finite
// logits. Reduction explicitly resolves equal values by index, including zeros.
kernel void vxt_greedy_exact(device const float* logits [[buffer(0)]],
    device uint* result [[buffer(1)]], constant uint& n [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint ids[256];
    threadgroup uint invalid[256];
    uint nonfinite = 0u;
    float best = -INFINITY;
    uint id = 0u;
    for (uint i = tid; i < n; i += 256u) {
        float v = logits[i];
        nonfinite |= !isfinite(v);
        if (v > best || (v == best && i < id)) { best = v; id = i; }
    }
    values[tid] = best;
    ids[tid] = id;
    invalid[tid] = nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride; stride >>= 1u) {
        if (tid < stride) {
            invalid[tid] |= invalid[tid + stride];
            float v = values[tid + stride];
            uint i = ids[tid + stride];
            if (v > values[tid] || (v == values[tid] && i < ids[tid])) {
                values[tid] = v; ids[tid] = i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) { result[0] = ids[0]; result[1] = invalid[0]; }
}
