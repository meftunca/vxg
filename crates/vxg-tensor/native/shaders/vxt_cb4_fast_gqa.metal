// ============================================================================
// vxt_cb4_fast_gqa.metal — Single-Dispatch Fused M=1 GQA & Batched Prefill
// DeltaNet / GQA Kernels (Eliminates 1,344 Per-Token Host Dispatches in Prefill!)
// ============================================================================

struct Cb4FastDnetBatchParams {
    uint n_kh, n_vh, dk, dv, conv_ch, k_sz, batch_m, qkv_z_stride, conv_stride, ab_stride, ab_half_stride, padded_inner;
    float eps, _pad0, _pad1, _pad2;
};

struct Cb4FastGqaBatchParams {
    uint nh, nkv, hd, pos_base, rope_dim, flags, batch_m, qkv_z_stride, q_rows, kv_dim, padded_att;
    float rope_freq_base, eps;
    uint rope_pos_base, mrope_s1, mrope_s2;
};

// 1. Batched Causal Conv1D + SiLU across `batch_m` tokens in 1 single dispatch (keeps history in regs!)
kernel void vxt_cb4_dnet_conv_batch(
    device const float* qkv_z           [[buffer(0)]],
    device const float* ssm_conv1d      [[buffer(1)]],
    device float* conv_state            [[buffer(2)]],
    device float* conv_silu_batch       [[buffer(3)]],
    constant Cb4FastDnetBatchParams& p  [[buffer(4)]],
    uint c                              [[thread_position_in_grid]]
) {
    if (c >= p.conv_ch) return;
    uint k_sz = p.k_sz;
    uint hist = (k_sz > 0u) ? (k_sz - 1u) : 0u;
    device const float* w = ssm_conv1d + c * k_sz;

    float st[8];
    for (uint t = 0u; t < hist; ++t) {
        st[t] = conv_state[t * p.conv_ch + c];
    }

    for (uint m = 0u; m < p.batch_m; ++m) {
        float x_c = qkv_z[ulong(m) * ulong(p.qkv_z_stride) + ulong(c)];
        float acc = 0.0f;
        for (uint t = 0u; t < hist; ++t) {
            acc = fma(st[t], w[t], acc);
        }
        acc = fma(x_c, w[hist], acc);
        if (hist > 0u) {
            for (uint t = 0u; t + 1u < hist; ++t) {
                st[t] = st[t + 1u];
            }
            st[hist - 1u] = x_c;
        }
        conv_silu_batch[ulong(m) * ulong(p.conv_stride) + ulong(c)] = acc / (1.0f + exp(-acc));
    }

    for (uint t = 0u; t < hist; ++t) {
        conv_state[t * p.conv_ch + c] = st[t];
    }
}

// 2. Batched FP32 DeltaNet State Update across `batch_m` tokens in 1 single dispatch (0-ULP exact!)
kernel void vxt_cb4_dnet_heads_batch(
    device const float* conv_silu_batch [[buffer(0)]],
    device const float* qkv_z           [[buffer(1)]],
    device const float* ab_batch        [[buffer(2)]],
    device const float* ssm_a           [[buffer(3)]],
    device const float* ssm_dt_bias     [[buffer(4)]],
    device const float* ssm_norm        [[buffer(5)]],
    device float* ssm_state             [[buffer(6)]],
    device float* y_concat              [[buffer(7)]],
    constant Cb4FastDnetBatchParams& p  [[buffer(8)]],
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
    device float* sh_base = ssm_state + ulong(h) * ulong(dv) * ulong(dk);

    threadgroup float q_sh[512];
    threadgroup float k_sh[512];
    threadgroup float o_sh[512];
    threadgroup float sg_red[8];

    for (uint m = 0u; m < p.batch_m; ++m) {
        device const float* cs = conv_silu_batch + ulong(m) * ulong(p.conv_stride);
        device const float* q_src = cs + hk * dk;
        device const float* k_src = cs + qk_dim + hk * dk;
        device const float* v_h = cs + 2u * qk_dim + h * dv;

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

        device const float* ab_m = ab_batch + ulong(m) * ulong(p.ab_stride);
        float beta = 1.0f / (1.0f + exp(-ab_m[p.ab_half_stride + h]));
        float x_sp = ab_m[h] + ssm_dt_bias[h];
        float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? exp(x_sp) : log(1.0f + exp(x_sp)));
        float decay = exp(sp * ssm_a[h]);

        if (dk == 128u) {
            uint i0 = lane, i1 = lane + 32u, i2 = lane + 64u, i3 = lane + 96u;
            float k0 = k_sh[i0], k1 = k_sh[i1], k2 = k_sh[i2], k3 = k_sh[i3];
            float q0 = q_sh[i0], q1 = q_sh[i1], q2 = q_sh[i2], q3 = q_sh[i3];
            for (uint jv = sg_id; jv < dv; jv += 8u) {
                device float* row = sh_base + (jv << 7u);
                float r0 = row[i0] * decay, r1 = row[i1] * decay;
                float r2 = row[i2] * decay, r3 = row[i3] * decay;
                float sk = simd_sum(fma(r3, k3, fma(r2, k2, fma(r1, k1, r0 * k0))));
                float d_jv = (v_h[jv] - sk) * beta;
                r0 = fma(d_jv, k0, r0); r1 = fma(d_jv, k1, r1);
                r2 = fma(d_jv, k2, r2); r3 = fma(d_jv, k3, r3);
                row[i0] = r0; row[i1] = r1; row[i2] = r2; row[i3] = r3;
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
        device const float* z_h = qkv_z + ulong(m) * ulong(p.qkv_z_stride) + p.conv_ch + h * dv;
        device float* dst_m = y_concat + ulong(m) * ulong(p.padded_inner);
        device float* dst_h = dst_m + h * dv;
        for (uint jv = tid; jv < dv; jv += 256u) {
            float zv = z_h[jv];
            dst_h[jv] = (o_sh[jv] * o_scale * ssm_norm[jv]) * (zv / (1.0f + exp(-zv)));
        }
        if (h == 0u && p.n_vh * dv < p.padded_inner) {
            for (uint col = p.n_vh * dv + tid; col < p.padded_inner; col += 256u) {
                dst_m[col] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline void cb4_gqa_norm_rope_sg(
    threadgroup float* h_buf,
    device const float* norm_w,
    bool has_norm,
    bool neox,
    uint hd,
    uint rope_dim,
    uint3 pos_thw,
    uint mrope_s1,
    uint mrope_s2,
    float rope_freq_base,
    float eps,
    uint lane
) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (has_norm) {
        float sq = 0.0f;
        for (uint d = lane; d < hd; d += 32u) {
            float v = h_buf[d];
            sq = fma(v, v, sq);
        }
        sq = simd_sum(sq);
        float sc = rsqrt(sq / float(hd) + eps);
        for (uint d = lane; d < hd; d += 32u) {
            h_buf[d] *= sc * norm_w[d];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint rdim = min(rope_dim, hd);
    uint half_r = rdim >> 1u;
    uint h_lim = mrope_s1 * 3u;
    uint w_lim = mrope_s2 * 3u;
    for (uint i = lane; i < half_r; i += 32u) {
        uint rpos = (i < h_lim && (i % 3u) == 1u) ? pos_thw.y
                  : ((i < w_lim && (i % 3u) == 2u) ? pos_thw.z : pos_thw.x);
        float freq = pow(rope_freq_base, -2.0f * float(i) / float(rdim));
        float theta = float(rpos) * freq;
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
}

// 3. Single-Dispatch Fused M=1 GQA Prep + Flash-Decoding Attention (0-ULP exact!)
kernel void vxt_cb4_gqa_fused_m1(
    device const float* qkv_z           [[buffer(0)]],
    device const float* q_norm          [[buffer(1)]],
    device const float* k_norm          [[buffer(2)]],
    device const float* q_bias          [[buffer(3)]],
    device const float* k_bias          [[buffer(4)]],
    device const float* v_bias          [[buffer(5)]],
    device float* k_cache               [[buffer(6)]],
    device float* v_cache               [[buffer(7)]],
    device float* att_cat               [[buffer(8)]],
    constant Cb4FastGqaBatchParams& p   [[buffer(9)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    if (h >= p.nh) return;

    uint hd = p.hd;
    uint gqa_mul = p.nh / p.nkv;
    uint kv_h = h / gqa_mul;
    uint pos = p.pos_base;
    uint3 rpos = uint3(p.rope_pos_base, p.rope_pos_base, p.rope_pos_base);
    uint seq_len = pos + 1u;
    ulong kv_stride = ulong(p.nkv * hd);
    bool gated_q = (p.flags & 1u) != 0u;
    bool neox = (p.flags & 2u) != 0u;
    bool has_qn = (p.flags & 4u) != 0u;
    bool has_kn = (p.flags & 8u) != 0u;
    bool has_qb = (p.flags & 16u) != 0u;
    bool has_kb = (p.flags & 32u) != 0u;
    bool has_vb = (p.flags & 64u) != 0u;

    threadgroup float q_sh[512];
    threadgroup float qg_sh[512];
    threadgroup float k_last[512];
    threadgroup float v_last[512];
    threadgroup float sg_m[8];
    threadgroup float sg_l[8];
    threadgroup float sg_acc[8 * 512];

    if (sg_id == 0u) {
        if (gated_q) {
            uint base = h * 2u * hd;
            for (uint d = lane; d < hd; d += 32u) {
                q_sh[d] = qkv_z[base + d] + (has_qb ? q_bias[base + d] : 0.0f);
                qg_sh[d] = qkv_z[base + hd + d] + (has_qb ? q_bias[base + hd + d] : 0.0f);
            }
        } else {
            uint base = h * hd;
            for (uint d = lane; d < hd; d += 32u) {
                q_sh[d] = qkv_z[base + d] + (has_qb ? q_bias[base + d] : 0.0f);
            }
        }
        cb4_gqa_norm_rope_sg(q_sh, q_norm, has_qn, neox, hd, p.rope_dim, rpos, 0u, 0u, p.rope_freq_base, p.eps, lane);
    } else if (sg_id == 1u) {
        device const float* k_cur = qkv_z + p.q_rows;
        device const float* v_cur = k_cur + p.kv_dim;
        uint base = kv_h * hd;
        ulong cache_off = ulong(pos) * kv_stride + ulong(base);
        bool write_kv = ((h % gqa_mul) == 0u);
        for (uint d = lane; d < hd; d += 32u) {
            float vv = v_cur[base + d] + (has_vb ? v_bias[base + d] : 0.0f);
            k_last[d] = k_cur[base + d] + (has_kb ? k_bias[base + d] : 0.0f);
            v_last[d] = vv;
            if (write_kv) v_cache[cache_off + d] = vv;
        }
        cb4_gqa_norm_rope_sg(k_last, k_norm, has_kn, neox, hd, p.rope_dim, rpos, 0u, 0u, p.rope_freq_base, p.eps, lane);
        if (write_kv) {
            for (uint d = lane; d < hd; d += 32u) {
                k_cache[cache_off + d] = k_last[d];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(float(hd));
    uint hd_chunks = (hd + 31u) >> 5u;
    float acc[16];
    for (uint c = 0u; c < hd_chunks; ++c) acc[c] = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint t = sg_id; t < seq_len; t += 8u) {
        float dot = 0.0f;
        if (t == pos) {
            for (uint c = 0u; c < hd_chunks; ++c) {
                uint d = (c << 5u) + lane;
                if (d < hd) dot = fma(q_sh[d], k_last[d], dot);
            }
        } else {
            device const float* kh = k_cache + ulong(t) * kv_stride + ulong(kv_h * hd);
            for (uint c = 0u; c < hd_chunks; ++c) {
                uint d = (c << 5u) + lane;
                if (d < hd) dot = fma(q_sh[d], kh[d], dot);
            }
        }
        float s = simd_sum(dot) * scale;
        float m_new = max(m_i, s);
        float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
        float w = exp(s - m_new);
        l_i = fma(l_i, alpha, w);
        m_i = m_new;
        if (t == pos) {
            for (uint c = 0u; c < hd_chunks; ++c) {
                uint d = (c << 5u) + lane;
                if (d < hd) acc[c] = fma(w, v_last[d], acc[c] * alpha);
            }
        } else {
            device const float* vh = v_cache + ulong(t) * kv_stride + ulong(kv_h * hd);
            for (uint c = 0u; c < hd_chunks; ++c) {
                uint d = (c << 5u) + lane;
                if (d < hd) acc[c] = fma(w, vh[d], acc[c] * alpha);
            }
        }
    }

    if (lane == 0u) {
        sg_m[sg_id] = m_i;
        sg_l[sg_id] = l_i;
    }
    for (uint c = 0u; c < hd_chunks; ++c) {
        uint d = (c << 5u) + lane;
        if (d < hd) sg_acc[sg_id * 512u + d] = acc[c];
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

    for (uint d = tid; d < hd; d += 256u) {
        float out_d = 0.0f;
        for (uint sg = 0u; sg < 8u; ++sg) {
            out_d = fma(sg_acc[sg * 512u + d], w_sg[sg], out_d);
        }
        out_d *= inv_l;
        if (gated_q) {
            float g = qg_sh[d];
            out_d *= 1.0f / (1.0f + exp(-g));
        }
        att_cat[h * hd + d] = out_d;
    }
    if (h == 0u && p.nh * hd < p.padded_att) {
        for (uint col = p.nh * hd + tid; col < p.padded_att; col += 256u) {
            att_cat[col] = 0.0f;
        }
    }
}

// 4. Batched GQA Prep across `batch_m` tokens in 1 single dispatch (tg_pos.y = m)
kernel void vxt_cb4_gqa_prep_batch(
    device const float* qkv_z           [[buffer(0)]],
    device const float* q_norm          [[buffer(1)]],
    device const float* k_norm          [[buffer(2)]],
    device const float* q_bias          [[buffer(3)]],
    device const float* k_bias          [[buffer(4)]],
    device const float* v_bias          [[buffer(5)]],
    device float* q_cur_out             [[buffer(6)]],
    device float* q_gate_out            [[buffer(7)]],
    device float* k_cache               [[buffer(8)]],
    device float* v_cache               [[buffer(9)]],
    constant Cb4FastGqaBatchParams& p   [[buffer(10)]],
    device const uint4* mrope_pos       [[buffer(11)]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint idx = tg_pos.x;
    uint m = tg_pos.y;
    if (m >= p.batch_m) return;

    uint hd = p.hd;
    uint pos = p.pos_base + m;
    uint rp = p.rope_pos_base + m;
    bool has_mrope = (p.flags & 128u) != 0u;
    uint3 rpos = has_mrope ? mrope_pos[m].xyz : uint3(rp, rp, rp);
    uint ms1 = has_mrope ? p.mrope_s1 : 0u;
    uint ms2 = has_mrope ? p.mrope_s2 : 0u;
    uint att_dim = p.nh * hd;
    bool gated_q = (p.flags & 1u) != 0u;
    bool neox = (p.flags & 2u) != 0u;
    bool has_qn = (p.flags & 4u) != 0u;
    bool has_kn = (p.flags & 8u) != 0u;
    bool has_qb = (p.flags & 16u) != 0u;
    bool has_kb = (p.flags & 32u) != 0u;
    bool has_vb = (p.flags & 64u) != 0u;

    device const float* q_raw = qkv_z + ulong(m) * ulong(p.qkv_z_stride);
    device const float* k_cur = q_raw + p.q_rows;
    device const float* v_cur = k_cur + p.kv_dim;

    threadgroup float h_buf[512];
    if (idx < p.nh) {
        uint h = idx;
        device float* q_dst = q_cur_out + ulong(m) * ulong(att_dim) + h * hd;
        device float* qg_dst = q_gate_out + ulong(m) * ulong(att_dim) + h * hd;
        if (gated_q) {
            uint base = h * 2u * hd;
            for (uint d = lane; d < hd; d += 32u) {
                h_buf[d] = q_raw[base + d] + (has_qb ? q_bias[base + d] : 0.0f);
                qg_dst[d] = q_raw[base + hd + d] + (has_qb ? q_bias[base + hd + d] : 0.0f);
            }
        } else {
            uint base = h * hd;
            for (uint d = lane; d < hd; d += 32u) {
                h_buf[d] = q_raw[base + d] + (has_qb ? q_bias[base + d] : 0.0f);
            }
        }
        cb4_gqa_norm_rope_sg(h_buf, q_norm, has_qn, neox, hd, p.rope_dim, rpos, ms1, ms2, p.rope_freq_base, p.eps, lane);
        for (uint d = lane; d < hd; d += 32u) q_dst[d] = h_buf[d];
    } else if (idx < p.nh + p.nkv) {
        uint kh = idx - p.nh;
        uint base = kh * hd;
        ulong cache_off = ulong(pos) * ulong(p.nkv * hd) + ulong(base);
        for (uint d = lane; d < hd; d += 32u) {
            h_buf[d] = k_cur[base + d] + (has_kb ? k_bias[base + d] : 0.0f);
            v_cache[cache_off + d] = v_cur[base + d] + (has_vb ? v_bias[base + d] : 0.0f);
        }
        cb4_gqa_norm_rope_sg(h_buf, k_norm, has_kn, neox, hd, p.rope_dim, rpos, ms1, ms2, p.rope_freq_base, p.eps, lane);
        for (uint d = lane; d < hd; d += 32u) k_cache[cache_off + d] = h_buf[d];
    }
}

// 5. Batched Causal GQA Attention across `batch_m` tokens in 1 single dispatch (tg_pos.y = m)
kernel void vxt_cb4_gqa_attn_batch(
    device const float* q_cur           [[buffer(0)]],
    device const float* q_gate          [[buffer(1)]],
    device const float* k_cache         [[buffer(2)]],
    device const float* v_cache         [[buffer(3)]],
    device float* att_cat               [[buffer(4)]],
    constant Cb4FastGqaBatchParams& p   [[buffer(5)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]]
) {
    uint h = tg_pos.x;
    uint m = tg_pos.y;
    if (h >= p.nh || m >= p.batch_m) return;

    uint hd = p.hd;
    uint att_dim = p.nh * hd;
    uint kv_h = h / (p.nh / p.nkv);
    uint seq_len = p.pos_base + m + 1u;
    ulong kv_stride = ulong(p.nkv * hd);
    float scale = rsqrt(float(hd));
    bool gated_q = (p.flags & 1u) != 0u;

    device const float* q_m = q_cur + ulong(m) * ulong(att_dim);
    device const float* qg_m = q_gate + ulong(m) * ulong(att_dim);
    device float* dst_m = att_cat + ulong(m) * ulong(p.padded_att);

    threadgroup float q_sh[512];
    threadgroup float sg_m[8];
    threadgroup float sg_l[8];
    threadgroup float sg_acc[8 * 512];

    for (uint d = tid; d < hd; d += 256u) q_sh[d] = q_m[h * hd + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint hd_chunks = (hd + 31u) >> 5u;
    float acc[16];
    for (uint c = 0u; c < hd_chunks; ++c) acc[c] = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint t = sg_id; t < seq_len; t += 8u) {
        ulong base = ulong(t) * kv_stride + ulong(kv_h * hd);
        device const float* kh = k_cache + base;
        device const float* vh = v_cache + base;
        float dot = 0.0f;
        for (uint c = 0u; c < hd_chunks; ++c) {
            uint d = (c << 5u) + lane;
            if (d < hd) dot = fma(q_sh[d], kh[d], dot);
        }
        float s = simd_sum(dot) * scale;
        float m_new = max(m_i, s);
        float alpha = (l_i == 0.0f) ? 0.0f : exp(m_i - m_new);
        float w = exp(s - m_new);
        l_i = fma(l_i, alpha, w);
        m_i = m_new;
        for (uint c = 0u; c < hd_chunks; ++c) {
            uint d = (c << 5u) + lane;
            if (d < hd) acc[c] = fma(w, vh[d], acc[c] * alpha);
        }
    }

    if (lane == 0u) {
        sg_m[sg_id] = m_i;
        sg_l[sg_id] = l_i;
    }
    for (uint c = 0u; c < hd_chunks; ++c) {
        uint d = (c << 5u) + lane;
        if (d < hd) sg_acc[sg_id * 512u + d] = acc[c];
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

    for (uint d = tid; d < hd; d += 256u) {
        float out_d = 0.0f;
        for (uint sg = 0u; sg < 8u; ++sg) {
            out_d = fma(sg_acc[sg * 512u + d], w_sg[sg], out_d);
        }
        out_d *= inv_l;
        if (gated_q) {
            float g = qg_m[h * hd + d];
            out_d *= 1.0f / (1.0f + exp(-g));
        }
        dst_m[h * hd + d] = out_d;
    }
    if (h == 0u && att_dim < p.padded_att) {
        for (uint col = att_dim + tid; col < p.padded_att; col += 256u) {
            dst_m[col] = 0.0f;
        }
    }
}
