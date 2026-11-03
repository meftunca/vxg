#include <metal_stdlib>
using namespace metal;

// Apple Silicon Metal GEMV compute kernels:
// - Codebook modes (`CB4` & `CB3`) evaluate learned non-uniform cubic centroids directly in
//   4-wide `float4` SIMD FMA registers (`0` LUT gathers, `0` shared memory, `0` shuffles!).
// - Lossless modes (`BF16` & `FP16`) use 100% coalesced `uchar[32]` superblock bit-planes +
//   hardware `simd_prefix_exclusive_sum` escape indexing + native 32-bit `lut_lo / lut_hi` shifts!

struct Raw16Params {
    uint rows;
    uint cols;
    uint is_fp16;
    uint _pad;
};

kernel void vxt_gemv_raw16(
    device const ushort* weights [[buffer(0)]],
    device const float* x        [[buffer(1)]],
    device float* y              [[buffer(2)]],
    constant Raw16Params& p      [[buffer(3)]],
    uint sg_id                   [[simdgroup_index_in_threadgroup]],
    uint lane                    [[thread_index_in_simdgroup]],
    uint3 tg_pos                 [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 2u + sg_id;
    if (row >= p.rows) return;

    device const ushort* row_ptr = weights + ulong(row) * ulong(p.cols);
    uint vec4_cols = p.cols & ~127u;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    if (p.is_fp16 != 0u) {
        for (uint base = 0u; base < vec4_cols; base += 128u) {
            uint c = base + (lane << 2u);
            half4 wh = as_type<half4>(*(device const ushort4*)(row_ptr + c));
            float4 xv = *(device const float4*)(x + c);
            acc0 = fma(float(wh.x), xv.x, fma(float(wh.z), xv.z, acc0));
            acc1 = fma(float(wh.y), xv.y, fma(float(wh.w), xv.w, acc1));
        }
        for (uint c = vec4_cols + lane; c < p.cols; c += 32u) {
            acc0 = fma(float(as_type<half>(row_ptr[c])), x[c], acc0);
        }
    } else {
        for (uint base = 0u; base < vec4_cols; base += 128u) {
            uint c = base + (lane << 2u);
            ushort4 wu = *(device const ushort4*)(row_ptr + c);
            float4 xv = *(device const float4*)(x + c);
            acc0 = fma(as_type<float>(uint(wu.x) << 16u), xv.x,
                   fma(as_type<float>(uint(wu.z) << 16u), xv.z, acc0));
            acc1 = fma(as_type<float>(uint(wu.y) << 16u), xv.y,
                   fma(as_type<float>(uint(wu.w) << 16u), xv.w, acc1));
        }
        for (uint c = vec4_cols + lane; c < p.cols; c += 32u) {
            acc0 = fma(as_type<float>(uint(row_ptr[c]) << 16u), x[c], acc0);
        }
    }

    float sum = simd_sum(acc0 + acc1);
    if (lane == 0u) {
        y[row] = sum;
    }
}

struct LosslessParams {
    uint rows;
    uint padded_cols;
    uint is_fp16;
    uint _pad;
    uint lut[8];
};

inline uint lut8_u32(uint code, uint lo4, uint hi4) {
    uint sel = (code < 4u) ? lo4 : hi4;
    return (sel >> ((code & 3u) << 3u)) & 0xffu;
}

inline float decode_bf16_sm(uchar sm_b, uint exp_s) {
    uint sm = uint(sm_b);
    return as_type<float>(((sm & 0x80u) << 24u) | exp_s | ((sm & 0x7fu) << 16u));
}

kernel void vxt_gemv_lossless_bf16(
    device const uchar* base_bytes [[buffer(0)]],
    device const uint* code_words  [[buffer(1)]],
    device const uint* sb_esc_off  [[buffer(2)]],
    device const uchar* esc_bytes  [[buffer(3)]],
    device const float* x_pad      [[buffer(4)]],
    device float* y                [[buffer(5)]],
    constant LosslessParams& p     [[buffer(6)]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 2u + sg_id;
    if (row >= p.rows) return;

    uint sbs_per_row = p.padded_cols >> 8u;
    uint row_sb_base = row * sbs_per_row;

    // Two native 32-bit uniform scalar registers hold all 8 byte exponents (lut[7] = 0)!
    uint lut_lo = (p.lut[0] & 0xffu) | ((p.lut[1] & 0xffu) << 8u)
                | ((p.lut[2] & 0xffu) << 16u) | ((p.lut[3] & 0xffu) << 24u);
    uint lut_hi = (p.lut[4] & 0xffu) | ((p.lut[5] & 0xffu) << 8u)
                | ((p.lut[6] & 0xffu) << 16u);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint sb = 0u; sb < sbs_per_row; ++sb) {
        uint sb_idx = row_sb_base + sb;
        device const uint2* sb_base2 = (device const uint2*)(base_bytes + (ulong(sb_idx) << 8u));
        uint2 m8 = sb_base2[lane];
        uchar4 m0 = as_type<uchar4>(m8.x);
        uchar4 m1 = as_type<uchar4>(m8.y);

        device const float4* xp4 = (device const float4*)(x_pad + (sb << 8u) + (lane << 3u));
        float4 x0 = xp4[0];
        float4 x1 = xp4[1];

        // 100% coalesced 32-byte plane reads across all 32 lanes!
        device const uchar* sb_cb = (device const uchar*)(code_words + ulong(sb_idx) * 24ul);
        uint b0 = uint(sb_cb[lane]);
        uint b1 = uint(sb_cb[32u + lane]);
        uint b2 = uint(sb_cb[64u + lane]);
        uint esc8 = b0 & b1 & b2;

        uint c0 = (b0 & 1u) | ((b1 & 1u) << 1u) | ((b2 & 1u) << 2u);
        uint c1 = ((b0 >> 1u) & 1u) | (((b1 >> 1u) & 1u) << 1u) | (((b2 >> 1u) & 1u) << 2u);
        uint c2 = ((b0 >> 2u) & 1u) | (((b1 >> 2u) & 1u) << 1u) | (((b2 >> 2u) & 1u) << 2u);
        uint c3 = ((b0 >> 3u) & 1u) | (((b1 >> 3u) & 1u) << 1u) | (((b2 >> 3u) & 1u) << 2u);
        uint c4 = ((b0 >> 4u) & 1u) | (((b1 >> 4u) & 1u) << 1u) | (((b2 >> 4u) & 1u) << 2u);
        uint c5 = ((b0 >> 5u) & 1u) | (((b1 >> 5u) & 1u) << 1u) | (((b2 >> 5u) & 1u) << 2u);
        uint c6 = ((b0 >> 6u) & 1u) | (((b1 >> 6u) & 1u) << 1u) | (((b2 >> 6u) & 1u) << 2u);
        uint c7 = ((b0 >> 7u) & 1u) | (((b1 >> 7u) & 1u) << 1u) | (((b2 >> 7u) & 1u) << 2u);

        uint e0 = lut8_u32(c0, lut_lo, lut_hi) << 23u;
        uint e1 = lut8_u32(c1, lut_lo, lut_hi) << 23u;
        uint e2 = lut8_u32(c2, lut_lo, lut_hi) << 23u;
        uint e3 = lut8_u32(c3, lut_lo, lut_hi) << 23u;
        uint e4 = lut8_u32(c4, lut_lo, lut_hi) << 23u;
        uint e5 = lut8_u32(c5, lut_lo, lut_hi) << 23u;
        uint e6 = lut8_u32(c6, lut_lo, lut_hi) << 23u;
        uint e7 = lut8_u32(c7, lut_lo, lut_hi) << 23u;

        if (simd_any(esc8 != 0u)) {
            uint pre = uint(simd_prefix_exclusive_sum(ushort(popcount(esc8))));
            if (esc8 != 0u) {
                uint e_pos = sb_esc_off[sb_idx] + pre;
                if (c0 == 7u) e0 = uint(esc_bytes[e_pos++]) << 23u;
                if (c1 == 7u) e1 = uint(esc_bytes[e_pos++]) << 23u;
                if (c2 == 7u) e2 = uint(esc_bytes[e_pos++]) << 23u;
                if (c3 == 7u) e3 = uint(esc_bytes[e_pos++]) << 23u;
                if (c4 == 7u) e4 = uint(esc_bytes[e_pos++]) << 23u;
                if (c5 == 7u) e5 = uint(esc_bytes[e_pos++]) << 23u;
                if (c6 == 7u) e6 = uint(esc_bytes[e_pos++]) << 23u;
                if (c7 == 7u) e7 = uint(esc_bytes[e_pos++]) << 23u;
            }
        }

        acc0 = fma(decode_bf16_sm(m0.x, e0), x0.x, acc0);
        acc1 = fma(decode_bf16_sm(m0.y, e1), x0.y, acc1);
        acc0 = fma(decode_bf16_sm(m0.z, e2), x0.z, acc0);
        acc1 = fma(decode_bf16_sm(m0.w, e3), x0.w, acc1);
        acc0 = fma(decode_bf16_sm(m1.x, e4), x1.x, acc0);
        acc1 = fma(decode_bf16_sm(m1.y, e5), x1.y, acc1);
        acc0 = fma(decode_bf16_sm(m1.z, e6), x1.z, acc0);
        acc1 = fma(decode_bf16_sm(m1.w, e7), x1.w, acc1);
    }

    float sum = simd_sum(acc0 + acc1);
    if (lane == 0u) {
        y[row] = sum;
    }
}

inline float decode_fp16_parts(uchar lo, uint sm, uint h1, uint h0, uint k, uint exp_v) {
    ushort bits16 = ushort(((((sm >> k) & 1u) << 15u)
                          | (exp_v << 10u)
                          | (((h1 >> k) & 1u) << 9u)
                          | (((h0 >> k) & 1u) << 8u))
                          | uint(lo));
    return float(as_type<half>(bits16));
}

kernel void vxt_gemv_lossless_fp16(
    device const uchar* base_bytes [[buffer(0)]],
    device const uint* code_words  [[buffer(1)]],
    device const uint* sb_esc_off  [[buffer(2)]],
    device const uchar* esc_bytes  [[buffer(3)]],
    device const float* x_pad      [[buffer(4)]],
    device float* y                [[buffer(5)]],
    constant LosslessParams& p     [[buffer(6)]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 2u + sg_id;
    if (row >= p.rows) return;

    uint sbs_per_row = p.padded_cols >> 8u;
    uint row_sb_base = row * sbs_per_row;

    uint lut_lo = (p.lut[0] & 0x1fu) | ((p.lut[1] & 0x1fu) << 8u)
                | ((p.lut[2] & 0x1fu) << 16u) | ((p.lut[3] & 0x1fu) << 24u);
    uint lut_hi = (p.lut[4] & 0x1fu) | ((p.lut[5] & 0x1fu) << 8u)
                | ((p.lut[6] & 0x1fu) << 16u);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint sb = 0u; sb < sbs_per_row; ++sb) {
        uint sb_idx = row_sb_base + sb;
        device const uint2* sb_base2 = (device const uint2*)(base_bytes + (ulong(sb_idx) << 8u));
        uint2 m8 = sb_base2[lane];
        uchar4 m0 = as_type<uchar4>(m8.x);
        uchar4 m1 = as_type<uchar4>(m8.y);

        device const float4* xp4 = (device const float4*)(x_pad + (sb << 8u) + (lane << 3u));
        float4 x0 = xp4[0];
        float4 x1 = xp4[1];

        device const uchar* sb_cb = (device const uchar*)(code_words + ulong(sb_idx) * 48ul);
        uint h0 = uint(sb_cb[lane]);
        uint h1 = uint(sb_cb[32u + lane]);
        uint sm = uint(sb_cb[64u + lane]);
        uint bc0 = uint(sb_cb[96u + lane]);
        uint bc1 = uint(sb_cb[128u + lane]);
        uint bc2 = uint(sb_cb[160u + lane]);
        uint esc8 = bc0 & bc1 & bc2;

        uint c0 = (bc0 & 1u) | ((bc1 & 1u) << 1u) | ((bc2 & 1u) << 2u);
        uint c1 = ((bc0 >> 1u) & 1u) | (((bc1 >> 1u) & 1u) << 1u) | (((bc2 >> 1u) & 1u) << 2u);
        uint c2 = ((bc0 >> 2u) & 1u) | (((bc1 >> 2u) & 1u) << 1u) | (((bc2 >> 2u) & 1u) << 2u);
        uint c3 = ((bc0 >> 3u) & 1u) | (((bc1 >> 3u) & 1u) << 1u) | (((bc2 >> 3u) & 1u) << 2u);
        uint c4 = ((bc0 >> 4u) & 1u) | (((bc1 >> 4u) & 1u) << 1u) | (((bc2 >> 4u) & 1u) << 2u);
        uint c5 = ((bc0 >> 5u) & 1u) | (((bc1 >> 5u) & 1u) << 1u) | (((bc2 >> 5u) & 1u) << 2u);
        uint c6 = ((bc0 >> 6u) & 1u) | (((bc1 >> 6u) & 1u) << 1u) | (((bc2 >> 6u) & 1u) << 2u);
        uint c7 = ((bc0 >> 7u) & 1u) | (((bc1 >> 7u) & 1u) << 1u) | (((bc2 >> 7u) & 1u) << 2u);

        uint e0 = lut8_u32(c0, lut_lo, lut_hi);
        uint e1 = lut8_u32(c1, lut_lo, lut_hi);
        uint e2 = lut8_u32(c2, lut_lo, lut_hi);
        uint e3 = lut8_u32(c3, lut_lo, lut_hi);
        uint e4 = lut8_u32(c4, lut_lo, lut_hi);
        uint e5 = lut8_u32(c5, lut_lo, lut_hi);
        uint e6 = lut8_u32(c6, lut_lo, lut_hi);
        uint e7 = lut8_u32(c7, lut_lo, lut_hi);

        if (simd_any(esc8 != 0u)) {
            uint pre = uint(simd_prefix_exclusive_sum(ushort(popcount(esc8))));
            if (esc8 != 0u) {
                uint e_pos = sb_esc_off[sb_idx] + pre;
                if (c0 == 7u) e0 = uint(esc_bytes[e_pos++]);
                if (c1 == 7u) e1 = uint(esc_bytes[e_pos++]);
                if (c2 == 7u) e2 = uint(esc_bytes[e_pos++]);
                if (c3 == 7u) e3 = uint(esc_bytes[e_pos++]);
                if (c4 == 7u) e4 = uint(esc_bytes[e_pos++]);
                if (c5 == 7u) e5 = uint(esc_bytes[e_pos++]);
                if (c6 == 7u) e6 = uint(esc_bytes[e_pos++]);
                if (c7 == 7u) e7 = uint(esc_bytes[e_pos++]);
            }
        }

        acc0 = fma(decode_fp16_parts(m0.x, sm, h1, h0, 0u, e0), x0.x, acc0);
        acc1 = fma(decode_fp16_parts(m0.y, sm, h1, h0, 1u, e1), x0.y, acc1);
        acc0 = fma(decode_fp16_parts(m0.z, sm, h1, h0, 2u, e2), x0.z, acc0);
        acc1 = fma(decode_fp16_parts(m0.w, sm, h1, h0, 3u, e3), x0.w, acc1);
        acc0 = fma(decode_fp16_parts(m1.x, sm, h1, h0, 4u, e4), x1.x, acc0);
        acc1 = fma(decode_fp16_parts(m1.y, sm, h1, h0, 5u, e5), x1.y, acc1);
        acc0 = fma(decode_fp16_parts(m1.z, sm, h1, h0, 6u, e6), x1.z, acc0);
        acc1 = fma(decode_fp16_parts(m1.w, sm, h1, h0, 7u, e7), x1.w, acc1);
    }

    float sum = simd_sum(acc0 + acc1);
    if (lane == 0u) {
        y[row] = sum;
    }
}

struct CodebookParams {
    uint rows;
    uint padded_cols;
    uint bits;
    uint _pad;
    float poly[16];
};

inline float4 eval_cubic4(float4 u, float a0, float a1, float a2, float a3) {
    float4 u2 = u * u;
    return fma(u, fma(a3, u2, a1), fma(a2, u2, a0));
}

kernel void vxt_gemv_cb4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const uint* out_row_ptr  [[buffer(2)]],
    device const uint* outliers     [[buffer(3)]],
    device const float* x_pad       [[buffer(4)]],
    device float* y                 [[buffer(5)]],
    constant CodebookParams& p      [[buffer(6)]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 2u + sg_id;
    if (row >= p.rows) return;

    uint groups_per_row = p.padded_cols >> 7u;
    uint steps = groups_per_row >> 1u;
    device const ushort* row_sc = scales_f16 + ulong(row) * ulong(groups_per_row);
    device const uint* row_pw = packed_words + ulong(row) * ulong(groups_per_row) * 16ul;
    uint grp_sub = lane >> 4u;

    float a0 = p.poly[0];
    float a1 = p.poly[1];
    float a2 = p.poly[2];
    float a3 = p.poly[3];

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint step = 0u; step < steps; ++step) {
        uint word = row_pw[(step << 5u) + lane];
        float sc = float(as_type<half>(row_sc[(step << 1u) + grp_sub]));

        device const float4* xp4 = (device const float4*)(x_pad + (step << 8u) + (lane << 3u));
        float4 x0 = xp4[0];
        float4 x1 = xp4[1];

        float4 u_lo = float4(uint4(word & 0x0fu, (word >> 4u) & 0x0fu,
                                   (word >> 8u) & 0x0fu, (word >> 12u) & 0x0fu)) - 7.5f;
        float4 u_hi = float4(uint4((word >> 16u) & 0x0fu, (word >> 20u) & 0x0fu,
                                   (word >> 24u) & 0x0fu, (word >> 28u) & 0x0fu)) - 7.5f;

        float4 w_lo = eval_cubic4(u_lo, a0, a1, a2, a3);
        float4 w_hi = eval_cubic4(u_hi, a0, a1, a2, a3);

        acc0 = fma(sc, dot(w_lo, x0), acc0);
        acc1 = fma(sc, dot(w_hi, x1), acc1);
    }

    uint os = out_row_ptr[row];
    uint oe = out_row_ptr[row + 1u];
    for (uint k = os + lane; k < oe; k += 32u) {
        uint pk = outliers[k];
        float val = float(as_type<half>(ushort(pk >> 16u)));
        acc0 = fma(val, x_pad[pk & 0xffffu], acc0);
    }

    float sum = simd_sum(acc0 + acc1);
    if (lane == 0u) {
        y[row] = sum;
    }
}

kernel void vxt_gemv_cb3(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const uint* out_row_ptr  [[buffer(2)]],
    device const uint* outliers     [[buffer(3)]],
    device const float* x_pad       [[buffer(4)]],
    device float* y                 [[buffer(5)]],
    constant CodebookParams& p      [[buffer(6)]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 2u + sg_id;
    if (row >= p.rows) return;

    uint groups_per_row = p.padded_cols >> 7u;
    uint steps = groups_per_row >> 1u;
    device const ushort* row_sc = scales_f16 + ulong(row) * ulong(groups_per_row);
    device const uint* row_pw = packed_words + ulong(row) * ulong(groups_per_row) * 12ul;
    uint grp_sub = lane >> 4u;

    float a0 = p.poly[0];
    float a1 = p.poly[1];
    float a2 = p.poly[2];
    float a3 = p.poly[3];

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint step = 0u; step < steps; ++step) {
        device const uint* p24 = row_pw + step * 24u;
        uint lo2 = uint(((device const ushort*)p24)[lane]);
        uint hi1 = uint(((device const uchar*)(p24 + 16u))[lane]);
        float sc = float(as_type<half>(row_sc[(step << 1u) + grp_sub]));

        device const float4* xp4 = (device const float4*)(x_pad + (step << 8u) + (lane << 3u));
        float4 x0 = xp4[0];
        float4 x1 = xp4[1];

        float4 u_lo = float4(uint4(
            (lo2 & 3u) | ((hi1 & 1u) << 2u),
            ((lo2 >> 2u) & 3u) | (((hi1 >> 1u) & 1u) << 2u),
            ((lo2 >> 4u) & 3u) | (((hi1 >> 2u) & 1u) << 2u),
            ((lo2 >> 6u) & 3u) | (((hi1 >> 3u) & 1u) << 2u)
        )) - 3.5f;
        float4 u_hi = float4(uint4(
            ((lo2 >> 8u) & 3u) | (((hi1 >> 4u) & 1u) << 2u),
            ((lo2 >> 10u) & 3u) | (((hi1 >> 5u) & 1u) << 2u),
            ((lo2 >> 12u) & 3u) | (((hi1 >> 6u) & 1u) << 2u),
            ((lo2 >> 14u) & 3u) | (((hi1 >> 7u) & 1u) << 2u)
        )) - 3.5f;

        float4 w_lo = eval_cubic4(u_lo, a0, a1, a2, a3);
        float4 w_hi = eval_cubic4(u_hi, a0, a1, a2, a3);

        acc0 = fma(sc, dot(w_lo, x0), acc0);
        acc1 = fma(sc, dot(w_hi, x1), acc1);
    }

    uint os = out_row_ptr[row];
    uint oe = out_row_ptr[row + 1u];
    for (uint k = os + lane; k < oe; k += 32u) {
        uint pk = outliers[k];
        float val = float(as_type<half>(ushort(pk >> 16u)));
        acc0 = fma(val, x_pad[pk & 0xffffu], acc0);
    }

    float sum = simd_sum(acc0 + acc1);
    if (lane == 0u) {
        y[row] = sum;
    }
}

struct Q4_0Params {
    uint rows;
    uint padded_cols;
};

kernel void vxt_gemv_q4_0(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* qs_words     [[buffer(1)]],
    device const float* x_pad       [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant Q4_0Params& p          [[buffer(4)]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 2u + sg_id;
    if (row >= p.rows) return;

    uint bpr = p.padded_cols >> 5u;
    uint b8_count = bpr >> 3u;
    device const ushort* row_sc = scales_f16 + ulong(row) * ulong(bpr);
    device const uint* row_qs = qs_words + ulong(row) * ulong(bpr) * 4ul;

    uint blk_sub = lane >> 2u;
    uint j0 = (lane & 3u) << 2u;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint b8 = 0u; b8 < b8_count; ++b8) {
        uint word = row_qs[(b8 << 5u) + lane];
        uint b_rel = (b8 << 3u) + blk_sub;
        float d = float(as_type<half>(row_sc[b_rel]));
        uint c0 = (b_rel << 5u) + j0;

        float4 x_lo = *(device const float4*)(x_pad + c0);
        float4 x_hi = *(device const float4*)(x_pad + c0 + 16u);

        float s_lo = float(int(word & 0x0fu) - 8) * x_lo.x
                   + float(int((word >> 8u) & 0x0fu) - 8) * x_lo.y
                   + float(int((word >> 16u) & 0x0fu) - 8) * x_lo.z
                   + float(int((word >> 24u) & 0x0fu) - 8) * x_lo.w;
        float s_hi = float(int((word >> 4u) & 0x0fu) - 8) * x_hi.x
                   + float(int((word >> 12u) & 0x0fu) - 8) * x_hi.y
                   + float(int((word >> 20u) & 0x0fu) - 8) * x_hi.z
                   + float(int((word >> 28u) & 0x0fu) - 8) * x_hi.w;
        acc0 = fma(d, s_lo, acc0);
        acc1 = fma(d, s_hi, acc1);
    }

    uint byte_idx = lane & 15u;
    bool hi_nib = lane >= 16u;
    for (uint b = (b8_count << 3u); b < bpr; ++b) {
        float d = float(as_type<half>(row_sc[b]));
        device const uchar* qs = (device const uchar*)(row_qs + (b << 2u));
        uchar bv = qs[byte_idx];
        int q = int(hi_nib ? (bv >> 4u) : (bv & 0x0fu)) - 8;
        acc0 = fma(d * float(q), x_pad[(b << 5u) + lane], acc0);
    }

    float sum = simd_sum(acc0 + acc1);
    if (lane == 0u) {
        y[row] = sum;
    }
}

kernel void vxt_fwht32_vec(
    device const float* x_pad      [[buffer(0)]],
    device float* x_rot            [[buffer(1)]],
    constant uint& padded_cols     [[buffer(2)]],
    uint sg_id                     [[simdgroup_index_in_threadgroup]],
    uint lane                      [[thread_index_in_simdgroup]],
    uint3 tg_pos                   [[threadgroup_position_in_grid]]
) {
    uint blk = tg_pos.x * 2u + sg_id;
    uint base = blk << 5u;
    if (base >= padded_cols) return;

    float v = x_pad[base + lane];
    for (uint mask = 1u; mask < 32u; mask <<= 1u) {
        float partner = simd_shuffle_xor(v, ushort(mask));
        v = ((lane & mask) == 0u) ? (v + partner) : (partner - v);
    }
    x_rot[base + lane] = v * 0.17677669529663689f;
}

kernel void vxt_gemv_wht2(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const uint* out_row_ptr  [[buffer(2)]],
    device const uint* outliers     [[buffer(3)]],
    device const float* x_raw_pad   [[buffer(4)]],
    device const float* x_rot_pad   [[buffer(5)]],
    device float* y                 [[buffer(6)]],
    constant CodebookParams& p      [[buffer(7)]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    uint row = tg_pos.x * 2u + sg_id;
    if (row >= p.rows) return;

    uint groups_per_row = p.padded_cols >> 7u;
    uint steps = groups_per_row >> 1u;
    device const ushort* row_sc = scales_f16 + ulong(row) * ulong(groups_per_row);
    device const uint* row_pw = packed_words + ulong(row) * ulong(steps) * 16ul;
    uint grp_sub = lane >> 4u;

    float a0 = p.poly[0];
    float a1 = p.poly[1];
    float a2 = p.poly[2];
    float a3 = p.poly[3];

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint step = 0u; step < steps; ++step) {
        uint code16 = uint(((device const ushort*)(row_pw + (step << 4u)))[lane]);
        float sc = float(as_type<half>(row_sc[(step << 1u) + grp_sub]));

        device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
        float4 x0 = xp4[0];
        float4 x1 = xp4[1];

        float4 u_lo = float4(uint4(code16 & 3u, (code16 >> 2u) & 3u,
                                   (code16 >> 4u) & 3u, (code16 >> 6u) & 3u)) - 1.5f;
        float4 u_hi = float4(uint4((code16 >> 8u) & 3u, (code16 >> 10u) & 3u,
                                   (code16 >> 12u) & 3u, (code16 >> 14u) & 3u)) - 1.5f;

        float4 w_lo = eval_cubic4(u_lo, a0, a1, a2, a3);
        float4 w_hi = eval_cubic4(u_hi, a0, a1, a2, a3);

        acc0 = fma(sc, dot(w_lo, x0), acc0);
        acc1 = fma(sc, dot(w_hi, x1), acc1);
    }

    uint os = out_row_ptr[row];
    uint oe = out_row_ptr[row + 1u];
    for (uint k = os + lane; k < oe; k += 32u) {
        uint pk = outliers[k];
        float val = float(as_type<half>(ushort(pk >> 16u)));
        acc0 = fma(val, x_raw_pad[pk & 0xffffu], acc0);
    }

    float sum = simd_sum(acc0 + acc1);
    if (lane == 0u) {
        y[row] = sum;
    }
}
