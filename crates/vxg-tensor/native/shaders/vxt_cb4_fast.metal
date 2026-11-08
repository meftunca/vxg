// ============================================================================
// vxt_cb4_fast.metal — Zero-Spill 32-bit SRAM LUT16 CB4+Outlier Matrix Kernels:
// - `vxt_cb4_gemv_nr4`: Tile-Major `NR=4` GEMV for 248K-row Vocabulary Head (342 GB/s)
// - `vxt_cb4_gemv_nr2`: Step-Major `NR=2` GEMV for Decoder Layer Projections (267 GB/s)
// - `vxt_cb4_ffn_gu_nr2_m1`: Fused `NR=2` `[Gate; Up] + SwiGLU` (4 rows/SG, zero spill)
// - `vxt_cb4_gemm_nr2`: Step-Major `NR=2 x BM=8` Multi-Slice Batched Prefill GEMM
// ============================================================================

struct Cb4FastGemvParams {
    uint rows;
    uint padded_cols;
    uint split0;
    uint split1;
    uint x_stride;
    uint y_stride;
    uint batch_m;
    uint flags; // bit0: add_res
    float poly[3][4];
};

struct Cb4FastFfnGuParams {
    uint n_ff;
    uint padded_ff;
    uint padded_cols;
    uint _pad;
    float poly_g[4];
    float poly_u[4];
};

inline float cb4_eval_c1(float u, float a0, float a1, float a2, float a3) {
    float u2 = u * u;
    return fma(u, fma(a3, u2, a1), fma(a2, u2, a0));
}

inline void cb4_decode_tg16(
    uint word, threadgroup const float* cb,
    thread float4& w_lo, thread float4& w_hi
) {
    w_lo = float4(
        cb[word & 15u], cb[(word >> 4u) & 15u],
        cb[(word >> 8u) & 15u], cb[(word >> 12u) & 15u]
    );
    w_hi = float4(
        cb[(word >> 16u) & 15u], cb[(word >> 20u) & 15u],
        cb[(word >> 24u) & 15u], cb[word >> 28u]
    );
}

inline void cb4_acc_tg16(
    uint word, ushort sc_h, float4 x0, float4 x1,
    threadgroup const float* cb, thread float& acc0, thread float& acc1
) {
    float4 w_lo, w_hi;
    cb4_decode_tg16(word, cb, w_lo, w_hi);
    float sc = float(as_type<half>(sc_h));
    acc0 = fma(sc, dot(w_lo, x0), acc0);
    acc1 = fma(sc, dot(w_hi, x1), acc1);
}

inline float cb4_outlier_sum(
    uint row, uint max_rows, uint lane, float acc0, float acc1,
    device const uint* out_row_ptr, device const uint* outliers, device const float* x_row
) {
    if (row >= max_rows) return 0.0f;
    uint os = out_row_ptr[row], oe = out_row_ptr[row + 1u];
    for (uint k = os + lane; k < oe; k += 32u) {
        uint pk = outliers[k];
        float val = float(as_type<half>(ushort(pk >> 16u)));
        acc0 = fma(val, x_row[pk & 0xffffu], acc0);
    }
    return simd_sum(acc0 + acc1);
}

inline void cb4_write_tile4_plain(
    device float* y, uint rows, uint row_base, float4 sum4, bool add_res
) {
    if (row_base + 3u < rows) {
        if (add_res) *(device float4*)(y + row_base) += sum4;
        else *(device float4*)(y + row_base) = sum4;
    } else if (add_res) {
        if (row_base < rows) y[row_base] += sum4.x;
        if (row_base + 1u < rows) y[row_base + 1u] += sum4.y;
        if (row_base + 2u < rows) y[row_base + 2u] += sum4.z;
    } else {
        if (row_base < rows) y[row_base] = sum4.x;
        if (row_base + 1u < rows) y[row_base + 1u] = sum4.y;
        if (row_base + 2u < rows) y[row_base + 2u] = sum4.z;
    }
}

inline void cb4_write_tile2_plain(
    device float* y, uint rows, uint row_base, float2 sum2, bool add_res
) {
    if (row_base + 1u < rows) {
        if (add_res) *(device float2*)(y + row_base) += sum2;
        else *(device float2*)(y + row_base) = sum2;
    } else if (row_base < rows) {
        if (add_res) y[row_base] += sum2.x;
        else y[row_base] = sum2.x;
    }
}

// 1. Tile-Major NR=4 GEMV for Large Vocabulary Head (keeps each 4 KB tile inside 1 VM page)
kernel void vxt_cb4_gemv_nr4(
    device const ushort4* tiled_scales  [[buffer(0)]],
    device const uint4* tiled_words     [[buffer(1)]],
    device const uint* out_row_ptr      [[buffer(2)]],
    device const uint* outliers         [[buffer(3)]],
    device const float* x_pad           [[buffer(4)]],
    device float* y                     [[buffer(5)]],
    constant Cb4FastGemvParams& p       [[buffer(6)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]],
    uint3 tg_size                       [[threads_per_threadgroup]]
) {
    threadgroup float c16[3][16];
    if (tid < 48u) {
        uint s = tid >> 4u, q = tid & 15u;
        c16[s][q] = cb4_eval_c1(float(q) - 7.5f, p.poly[s][0], p.poly[s][1], p.poly[s][2], p.poly[s][3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile = tg_pos.x * (tg_size.x >> 5u) + sg_id;
    uint row_sg = tile << 2u;
    if (row_sg >= p.rows) return;

    uint sub = (row_sg < p.split0) ? 0u : ((row_sg < p.split1) ? 1u : 2u);
    threadgroup const float* cb = c16[sub];
    uint groups_per_row = p.padded_cols >> 7u;
    uint steps = groups_per_row >> 1u;
    uint grp_sub = lane >> 4u;

    device const ushort4* tile_sc = tiled_scales + ulong(tile) * ulong(groups_per_row);
    device const uint4* tile_pw = tiled_words + ulong(tile) * ulong(steps) * 32ul;

    float a0_0 = 0.0f, a1_0 = 0.0f, a0_1 = 0.0f, a1_1 = 0.0f;
    float a0_2 = 0.0f, a1_2 = 0.0f, a0_3 = 0.0f, a1_3 = 0.0f;

    for (uint step = 0u; step < steps; ++step) {
        device const float4* xp4 = (device const float4*)(x_pad + (step << 8u) + (lane << 3u));
        float4 x0 = xp4[0], x1 = xp4[1];
        uint4 pw = tile_pw[(step << 5u) + lane];
        ushort4 sc4 = tile_sc[(step << 1u) + grp_sub];
        cb4_acc_tg16(pw.x, sc4.x, x0, x1, cb, a0_0, a1_0);
        cb4_acc_tg16(pw.y, sc4.y, x0, x1, cb, a0_1, a1_1);
        cb4_acc_tg16(pw.z, sc4.z, x0, x1, cb, a0_2, a1_2);
        cb4_acc_tg16(pw.w, sc4.w, x0, x1, cb, a0_3, a1_3);
    }

    float s0 = cb4_outlier_sum(row_sg + 0u, p.rows, lane, a0_0, a1_0, out_row_ptr, outliers, x_pad);
    float s1 = cb4_outlier_sum(row_sg + 1u, p.rows, lane, a0_1, a1_1, out_row_ptr, outliers, x_pad);
    float s2 = cb4_outlier_sum(row_sg + 2u, p.rows, lane, a0_2, a1_2, out_row_ptr, outliers, x_pad);
    float s3 = cb4_outlier_sum(row_sg + 3u, p.rows, lane, a0_3, a1_3, out_row_ptr, outliers, x_pad);
    if (lane == 0u) {
        cb4_write_tile4_plain(y, p.rows, row_sg, float4(s0, s1, s2, s3), (p.flags & 1u) != 0u);
    }
}

// 2. Step-Major NR=2 GEMV for Decoder Layer Matrices (eliminates 4096B bank aliasing + doubles waves)
kernel void vxt_cb4_gemv_nr2(
    device const ushort2* step_scales   [[buffer(0)]],
    device const uint2* step_words      [[buffer(1)]],
    device const uint* out_row_ptr      [[buffer(2)]],
    device const uint* outliers         [[buffer(3)]],
    device const float* x_pad           [[buffer(4)]],
    device float* y                     [[buffer(5)]],
    constant Cb4FastGemvParams& p       [[buffer(6)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]],
    uint3 tg_size                       [[threads_per_threadgroup]]
) {
    threadgroup float c16[3][16];
    if (tid < 48u) {
        uint s = tid >> 4u, q = tid & 15u;
        c16[s][q] = cb4_eval_c1(float(q) - 7.5f, p.poly[s][0], p.poly[s][1], p.poly[s][2], p.poly[s][3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile = tg_pos.x * (tg_size.x >> 5u) + sg_id;
    uint row_sg = tile << 1u;
    if (row_sg >= p.rows) return;

    uint sub = (row_sg < p.split0) ? 0u : ((row_sg < p.split1) ? 1u : 2u);
    threadgroup const float* cb = c16[sub];
    uint steps = p.padded_cols >> 8u;
    uint grp_sub = lane >> 4u;
    ulong num_tiles = ulong((p.rows + 1u) >> 1u);

    float a0_0 = 0.0f, a1_0 = 0.0f, a0_1 = 0.0f, a1_1 = 0.0f;
    for (uint step = 0u; step < steps; ++step) {
        device const float4* xp4 = (device const float4*)(x_pad + (step << 8u) + (lane << 3u));
        float4 x0 = xp4[0], x1 = xp4[1];
        ulong idx_base = ulong(step) * num_tiles + ulong(tile);
        uint2 pw = step_words[(idx_base << 5u) + lane];
        ushort2 sc2 = step_scales[(idx_base << 1u) + grp_sub];
        cb4_acc_tg16(pw.x, sc2.x, x0, x1, cb, a0_0, a1_0);
        cb4_acc_tg16(pw.y, sc2.y, x0, x1, cb, a0_1, a1_1);
    }

    float s0 = cb4_outlier_sum(row_sg + 0u, p.rows, lane, a0_0, a1_0, out_row_ptr, outliers, x_pad);
    float s1 = cb4_outlier_sum(row_sg + 1u, p.rows, lane, a0_1, a1_1, out_row_ptr, outliers, x_pad);
    if (lane == 0u) {
        cb4_write_tile2_plain(y, p.rows, row_sg, float2(s0, s1), (p.flags & 1u) != 0u);
    }
}

// 3. Fused M=1 NR=2 [ffn_gate; ffn_up] + SwiGLU (2 gate + 2 up = 4 rows/SG -> zero spill, 0-ULP!)
kernel void vxt_cb4_ffn_gu_nr2_m1(
    device const ushort2* step_scales   [[buffer(0)]],
    device const uint2* step_words      [[buffer(1)]],
    device const uint* out_row_ptr      [[buffer(2)]],
    device const uint* outliers         [[buffer(3)]],
    device const float* x_pad           [[buffer(4)]],
    device float* act_out               [[buffer(5)]],
    constant Cb4FastFfnGuParams& p      [[buffer(6)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]],
    uint3 tg_size                       [[threads_per_threadgroup]]
) {
    threadgroup float c16_g[16];
    threadgroup float c16_u[16];
    if (tid < 16u) {
        c16_g[tid] = cb4_eval_c1(float(tid) - 7.5f, p.poly_g[0], p.poly_g[1], p.poly_g[2], p.poly_g[3]);
    } else if (tid < 32u) {
        uint q = tid - 16u;
        c16_u[q] = cb4_eval_c1(float(q) - 7.5f, p.poly_u[0], p.poly_u[1], p.poly_u[2], p.poly_u[3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tiles_g = p.n_ff >> 1u;
    uint tile_g = tg_pos.x * (tg_size.x >> 5u) + sg_id;
    if (tile_g < tiles_g) {
        uint tile_u = tile_g + tiles_g;
        uint steps = p.padded_cols >> 8u;
        uint grp_sub = lane >> 4u;
        ulong num_tiles = ulong(p.n_ff);

        float g0_0 = 0.0f, g1_0 = 0.0f, g0_1 = 0.0f, g1_1 = 0.0f;
        float u0_0 = 0.0f, u1_0 = 0.0f, u0_1 = 0.0f, u1_1 = 0.0f;

        for (uint step = 0u; step < steps; ++step) {
            device const float4* xp4 = (device const float4*)(x_pad + (step << 8u) + (lane << 3u));
            float4 x0 = xp4[0], x1 = xp4[1];
            ulong sb = ulong(step) * num_tiles;
            uint2 pwg = step_words[((sb + ulong(tile_g)) << 5u) + lane];
            uint2 pwu = step_words[((sb + ulong(tile_u)) << 5u) + lane];
            ushort2 scg = step_scales[((sb + ulong(tile_g)) << 1u) + grp_sub];
            ushort2 scu = step_scales[((sb + ulong(tile_u)) << 1u) + grp_sub];
            cb4_acc_tg16(pwg.x, scg.x, x0, x1, c16_g, g0_0, g1_0);
            cb4_acc_tg16(pwg.y, scg.y, x0, x1, c16_g, g0_1, g1_1);
            cb4_acc_tg16(pwu.x, scu.x, x0, x1, c16_u, u0_0, u1_0);
            cb4_acc_tg16(pwu.y, scu.y, x0, x1, c16_u, u0_1, u1_1);
        }

        uint max_r = 2u * p.n_ff;
        uint rg = tile_g << 1u;
        uint ru = p.n_ff + rg;
        float g0 = cb4_outlier_sum(rg + 0u, max_r, lane, g0_0, g1_0, out_row_ptr, outliers, x_pad);
        float g1 = cb4_outlier_sum(rg + 1u, max_r, lane, g0_1, g1_1, out_row_ptr, outliers, x_pad);
        float u0 = cb4_outlier_sum(ru + 0u, max_r, lane, u0_0, u1_0, out_row_ptr, outliers, x_pad);
        float u1 = cb4_outlier_sum(ru + 1u, max_r, lane, u0_1, u1_1, out_row_ptr, outliers, x_pad);

        if (lane == 0u) {
            float2 gv = float2(g0, g1);
            float2 uv = float2(u0, u1);
            *(device float2*)(act_out + rg) = (gv / (1.0f + exp(-gv))) * uv;
        }
    }
    if (tg_pos.x == 0u && p.n_ff < p.padded_ff) {
        for (uint col = p.n_ff + tid; col < p.padded_ff; col += tg_size.x) {
            act_out[col] = 0.0f;
        }
    }
}

// 4. Multi-Slice Step-Major NR=2 x BM=8 Batched Prefill GEMM (8 tokens per weight read, 0-ULP!)
kernel void vxt_cb4_gemm_nr2(
    device const ushort2* step_scales   [[buffer(0)]],
    device const uint2* step_words      [[buffer(1)]],
    device const uint* out_row_ptr      [[buffer(2)]],
    device const uint* outliers         [[buffer(3)]],
    device const float* x_pad_base      [[buffer(4)]],
    device float* y_base                [[buffer(5)]],
    constant Cb4FastGemvParams& p       [[buffer(6)]],
    uint tid                            [[thread_index_in_threadgroup]],
    uint sg_id                          [[simdgroup_index_in_threadgroup]],
    uint lane                           [[thread_index_in_simdgroup]],
    uint3 tg_pos                        [[threadgroup_position_in_grid]],
    uint3 tg_size                       [[threads_per_threadgroup]]
) {
    threadgroup float c16[3][16];
    if (tid < 48u) {
        uint s = tid >> 4u, q = tid & 15u;
        c16[s][q] = cb4_eval_c1(float(q) - 7.5f, p.poly[s][0], p.poly[s][1], p.poly[s][2], p.poly[s][3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile = tg_pos.x * (tg_size.x >> 5u) + sg_id;
    uint row_sg = tile << 1u;
    uint m_base = tg_pos.y << 3u;
    if (row_sg >= p.rows || m_base >= p.batch_m) return;

    uint bm = min(8u, p.batch_m - m_base);
    ulong xs = ulong(p.x_stride);
    ulong ys = ulong(p.y_stride);
    device const float* x_pad = x_pad_base + ulong(m_base) * xs;
    device float* y = y_base + ulong(m_base) * ys;

    uint sub = (row_sg < p.split0) ? 0u : ((row_sg < p.split1) ? 1u : 2u);
    threadgroup const float* cb = c16[sub];
    uint steps = p.padded_cols >> 8u;
    uint grp_sub = lane >> 4u;
    ulong num_tiles = ulong((p.rows + 1u) >> 1u);

    float2 a0[8], a1[8];
    for (uint m = 0u; m < 8u; ++m) {
        a0[m] = float2(0.0f);
        a1[m] = float2(0.0f);
    }

    for (uint step = 0u; step < steps; ++step) {
        ulong idx_base = ulong(step) * num_tiles + ulong(tile);
        uint2 pw = step_words[(idx_base << 5u) + lane];
        float2 sc = float2(as_type<half2>(step_scales[(idx_base << 1u) + grp_sub]));
        float4 w0_lo, w0_hi, w1_lo, w1_hi;
        cb4_decode_tg16(pw.x, cb, w0_lo, w0_hi);
        cb4_decode_tg16(pw.y, cb, w1_lo, w1_hi);

        uint x_off = (step << 8u) + (lane << 3u);
        for (uint m = 0u; m < bm; ++m) {
            device const float4* xpm = (device const float4*)(x_pad + ulong(m) * xs + x_off);
            float4 xa = xpm[0], xb = xpm[1];
            a0[m] = fma(sc, float2(dot(w0_lo, xa), dot(w1_lo, xa)), a0[m]);
            a1[m] = fma(sc, float2(dot(w0_hi, xb), dot(w1_hi, xb)), a1[m]);
        }
    }

    for (uint r = 0u; r < 2u; ++r) {
        uint row = row_sg + r;
        if (row < p.rows) {
            uint os = out_row_ptr[row], oe = out_row_ptr[row + 1u];
            for (uint k = os + lane; k < oe; k += 32u) {
                uint pk = outliers[k];
                uint col = pk & 0xffffu;
                float val = float(as_type<half>(ushort(pk >> 16u)));
                if (r == 0u) {
                    for (uint m = 0u; m < bm; ++m) a0[m].x = fma(val, x_pad[ulong(m) * xs + col], a0[m].x);
                } else {
                    for (uint m = 0u; m < bm; ++m) a0[m].y = fma(val, x_pad[ulong(m) * xs + col], a0[m].y);
                }
            }
        }
    }

    bool add_res = (p.flags & 1u) != 0u;
    for (uint m = 0u; m < bm; ++m) {
        float2 sm = simd_sum(a0[m] + a1[m]);
        if (lane == 0u) {
            cb4_write_tile2_plain(y + ulong(m) * ys, p.rows, row_sg, sm, add_res);
        }
    }
}
