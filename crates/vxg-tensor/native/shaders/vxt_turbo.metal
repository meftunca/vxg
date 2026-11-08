// ============================================================================
// vxt_turbo.metal — Outlier-Free Sign-Flipped Walsh-Hadamard VQ Kernels
// (`VXT-Turbo2` 2.125 bpw Orth4D-16 & `VXT-Turbo3` 3.125 bpw Orth2D-16)
// Ultra-Compact 32-Bit `half2` Shared-Memory LUTs (`512 B` / `256 B`) + `uint2` Tiles
// ============================================================================

constant ushort4 ORTH4D_16_U16[16] = {
    ushort4(0x35c4u, 0x35e9u, 0x35cau, 0x35cbu),
    ushort4(0x3534u, 0x3cbeu, 0x377bu, 0x37b3u),
    ushort4(0x3cc3u, 0x3540u, 0x37a5u, 0x3757u),
    ushort4(0x374au, 0x3738u, 0x372bu, 0x3cd3u),
    ushort4(0x3757u, 0x372fu, 0x3cd8u, 0x3726u),
    ushort4(0x3c7cu, 0x3c7eu, 0x3741u, 0x379fu),
    ushort4(0x38afu, 0x38a2u, 0x3e16u, 0x3e2du),
    ushort4(0x3885u, 0x3e2cu, 0x3e2du, 0x38beu),
    ushort4(0x3e2cu, 0x38afu, 0x38bfu, 0x3e31u),
    ushort4(0x3e47u, 0x38cdu, 0x3e30u, 0x38beu),
    ushort4(0x38b5u, 0x3e30u, 0x38deu, 0x3e62u),
    ushort4(0x3868u, 0x4071u, 0x385cu, 0x3865u),
    ushort4(0x408du, 0x389fu, 0x385cu, 0x3885u),
    ushort4(0x38a9u, 0x3866u, 0x3893u, 0x4094u),
    ushort4(0x38b6u, 0x3885u, 0x4097u, 0x38d9u),
    ushort4(0x3f25u, 0x3f5eu, 0x3a22u, 0x3a06u),
};

constant ushort2 ORTH2D_16_U16[16] = {
    ushort2(0x3297u, 0x32aau),
    ushort2(0x390du, 0x33cfu),
    ushort2(0x331bu, 0x3932u),
    ushort2(0x3973u, 0x39fdu),
    ushort2(0x3c6cu, 0x33eeu),
    ushort2(0x33e9u, 0x3ca3u),
    ushort2(0x3cbcu, 0x3a20u),
    ushort2(0x3a26u, 0x3d4au),
    ushort2(0x3ec5u, 0x3473u),
    ushort2(0x34f5u, 0x3f17u),
    ushort2(0x3d68u, 0x3d72u),
    ushort2(0x3f4du, 0x3b2eu),
    ushort2(0x3c80u, 0x4022u),
    ushort2(0x4113u, 0x37aeu),
    ushort2(0x37abu, 0x4138u),
    ushort2(0x4052u, 0x3f10u),
};

constant ushort2 EC_ORTH2D_16_U16[16] = {
    ushort2(0x0000u, 0x0000u),
    ushort2(0x3981u, 0x346cu),
    ushort2(0x346cu, 0x3981u),
    ushort2(0x3b48u, 0x3b48u),
    ushort2(0x3d57u, 0x352au),
    ushort2(0x352au, 0x3d57u),
    ushort2(0x3e61u, 0x3c01u),
    ushort2(0x3c01u, 0x3e61u),
    ushort2(0x4014u, 0x35c7u),
    ushort2(0x35c7u, 0x4014u),
    ushort2(0x3edbu, 0x3edbu),
    ushort2(0x40ccu, 0x3cc9u),
    ushort2(0x3cc9u, 0x40ccu),
    ushort2(0x41c2u, 0x3726u),
    ushort2(0x3726u, 0x41c2u),
    ushort2(0x409eu, 0x409eu),
};

constant ushort4 ORTH4D_64_U16[64] = {
    ushort4(0x0000u, 0x0000u, 0x0000u, 0x0000u), ushort4(0x3b24u, 0x0000u, 0x0000u, 0x0000u), ushort4(0x0000u, 0x3b24u, 0x0000u, 0x0000u), ushort4(0x0000u, 0x0000u, 0x3b24u, 0x0000u),
    ushort4(0x0000u, 0x0000u, 0x0000u, 0x3b24u), ushort4(0x0000u, 0x3c6au, 0x3c6au, 0x3c6au), ushort4(0x3c6au, 0x0000u, 0x3c6au, 0x3c6au), ushort4(0x3c6au, 0x3c6au, 0x0000u, 0x3c6au),
    ushort4(0x3c6au, 0x3c6au, 0x3c6au, 0x0000u), ushort4(0x3645u, 0x3645u, 0x3645u, 0x3645u), ushort4(0x3ad0u, 0x3ad0u, 0x3ad0u, 0x3ad0u), ushort4(0x3dbau, 0x3dbau, 0x3dbau, 0x3dbau),
    ushort4(0x3dedu, 0x3603u, 0x3603u, 0x3603u), ushort4(0x3603u, 0x3dedu, 0x3603u, 0x3603u), ushort4(0x3603u, 0x3603u, 0x3dedu, 0x3603u), ushort4(0x3603u, 0x3603u, 0x3603u, 0x3dedu),
    ushort4(0x40b9u, 0x3602u, 0x3602u, 0x3602u), ushort4(0x3602u, 0x40b9u, 0x3602u, 0x3602u), ushort4(0x3602u, 0x3602u, 0x40b9u, 0x3602u), ushort4(0x3602u, 0x3602u, 0x3602u, 0x40b9u),
    ushort4(0x41b7u, 0x3a89u, 0x3a89u, 0x3a89u), ushort4(0x3a89u, 0x41b7u, 0x3a89u, 0x3a89u), ushort4(0x3a89u, 0x3a89u, 0x41b7u, 0x3a89u), ushort4(0x3a89u, 0x3a89u, 0x3a89u, 0x41b7u),
    ushort4(0x355au, 0x3e7bu, 0x3e7bu, 0x3e7bu), ushort4(0x3e7bu, 0x355au, 0x3e7bu, 0x3e7bu), ushort4(0x3e7bu, 0x3e7bu, 0x355au, 0x3e7bu), ushort4(0x3e7bu, 0x3e7bu, 0x3e7bu, 0x355au),
    ushort4(0x3b89u, 0x3b89u, 0x351au, 0x351au), ushort4(0x3b89u, 0x351au, 0x3b89u, 0x351au), ushort4(0x3b89u, 0x351au, 0x351au, 0x3b89u), ushort4(0x351au, 0x3b89u, 0x3b89u, 0x351au),
    ushort4(0x351au, 0x3b89u, 0x351au, 0x3b89u), ushort4(0x351au, 0x351au, 0x3b89u, 0x3b89u), ushort4(0x3d9bu, 0x3d9bu, 0x3a4du, 0x3a4du), ushort4(0x3d9bu, 0x3a4du, 0x3d9bu, 0x3a4du),
    ushort4(0x3d9bu, 0x3a4du, 0x3a4du, 0x3d9bu), ushort4(0x3a4du, 0x3d9bu, 0x3d9bu, 0x3a4du), ushort4(0x3a4du, 0x3d9bu, 0x3a4du, 0x3d9bu), ushort4(0x3a4du, 0x3a4du, 0x3d9bu, 0x3d9bu),
    ushort4(0x403fu, 0x403fu, 0x3780u, 0x3780u), ushort4(0x403fu, 0x3780u, 0x403fu, 0x3780u), ushort4(0x403fu, 0x3780u, 0x3780u, 0x403fu), ushort4(0x3780u, 0x403fu, 0x403fu, 0x3780u),
    ushort4(0x3780u, 0x403fu, 0x3780u, 0x403fu), ushort4(0x3780u, 0x3780u, 0x403fu, 0x403fu), ushort4(0x4029u, 0x4029u, 0x3c68u, 0x3c68u), ushort4(0x4029u, 0x3c68u, 0x4029u, 0x3c68u),
    ushort4(0x4029u, 0x3c68u, 0x3c68u, 0x4029u), ushort4(0x3c68u, 0x4029u, 0x4029u, 0x3c68u), ushort4(0x3c68u, 0x4029u, 0x3c68u, 0x4029u), ushort4(0x3c68u, 0x3c68u, 0x4029u, 0x4029u),
    ushort4(0x3f7cu, 0x3c98u, 0x3698u, 0x3698u), ushort4(0x3f7cu, 0x3698u, 0x3c98u, 0x3698u), ushort4(0x3f7cu, 0x3698u, 0x3698u, 0x3c98u), ushort4(0x3c98u, 0x3f7cu, 0x3698u, 0x3698u),
    ushort4(0x3698u, 0x3f7cu, 0x3c98u, 0x3698u), ushort4(0x3698u, 0x3f7cu, 0x3698u, 0x3c98u), ushort4(0x3c98u, 0x3698u, 0x3f7cu, 0x3698u), ushort4(0x3698u, 0x3c98u, 0x3f7cu, 0x3698u),
    ushort4(0x3698u, 0x3698u, 0x3f7cu, 0x3c98u), ushort4(0x3c98u, 0x3698u, 0x3698u, 0x3f7cu), ushort4(0x3698u, 0x3c98u, 0x3698u, 0x3f7cu), ushort4(0x3698u, 0x3698u, 0x3c98u, 0x3f7cu),
};

constant ushort4 SGN16_U16[16] = {
    ushort4(0x0000u, 0x0000u, 0x0000u, 0x0000u), ushort4(0x8000u, 0x0000u, 0x0000u, 0x0000u),
    ushort4(0x0000u, 0x8000u, 0x0000u, 0x0000u), ushort4(0x8000u, 0x8000u, 0x0000u, 0x0000u),
    ushort4(0x0000u, 0x0000u, 0x8000u, 0x0000u), ushort4(0x8000u, 0x0000u, 0x8000u, 0x0000u),
    ushort4(0x0000u, 0x8000u, 0x8000u, 0x0000u), ushort4(0x8000u, 0x8000u, 0x8000u, 0x0000u),
    ushort4(0x0000u, 0x0000u, 0x0000u, 0x8000u), ushort4(0x8000u, 0x0000u, 0x0000u, 0x8000u),
    ushort4(0x0000u, 0x8000u, 0x0000u, 0x8000u), ushort4(0x8000u, 0x8000u, 0x0000u, 0x8000u),
    ushort4(0x0000u, 0x0000u, 0x8000u, 0x8000u), ushort4(0x8000u, 0x0000u, 0x8000u, 0x8000u),
    ushort4(0x0000u, 0x8000u, 0x8000u, 0x8000u), ushort4(0x8000u, 0x8000u, 0x8000u, 0x8000u),
};

struct TurboParams {
    uint rows;
    uint padded_cols;
    uint cols;
    uint tiles_per_sg;
};

struct TurboRotParams {
    uint cols;
    uint padded_cols;
};

inline uint vxt_turbo_sign_mask(uint col) {
    uint h = col ^ 0x9e3779b9u;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return (h & 1u) << 31u;
}

inline float vxt_apply_col_sign(float v, uint col) {
    return (v == 0.0f) ? 0.0f : as_type<float>(as_type<uint>(v) ^ vxt_turbo_sign_mask(col));
}

inline float vxt_fwht32_lane(float v, uint lane) {
    for (uint mask = 1u; mask < 32u; mask <<= 1u) {
        float partner = simd_shuffle_xor(v, ushort(mask));
        v = ((lane & mask) == 0u) ? (v + partner) : (partner - v);
    }
    return v * 0.17677669529663689f;
}

kernel void vxt_turbo_fwht32_vec(
    device const float* x_in   [[buffer(0)]],
    device float* x_rot        [[buffer(1)]],
    constant TurboRotParams& p [[buffer(2)]],
    uint sg_id                 [[simdgroup_index_in_threadgroup]],
    uint lane                  [[thread_index_in_simdgroup]],
    uint3 tg_pos               [[threadgroup_position_in_grid]]
) {
    uint blk = tg_pos.x * 8u + sg_id;
    uint base = blk << 5u;
    if (base >= p.padded_cols) return;
    uint col = base + lane;
    float v = (col < p.cols) ? vxt_apply_col_sign(x_in[col], col) : 0.0f;
    x_rot[col] = vxt_fwht32_lane(v, lane);
}

inline void vxt_init_lut256(threadgroup half4* lut256, uint tid) {
    for (uint idx = tid; idx < 256u; idx += 128u) {
        ushort4 u = ORTH4D_16_U16[idx & 15u];
        u.x |= ushort((idx & 0x10u) << 11u);
        u.y |= ushort((idx & 0x20u) << 10u);
        u.z |= ushort((idx & 0x40u) << 9u);
        u.w |= ushort((idx & 0x80u) << 8u);
        lut256[idx] = as_type<half4>(u);
    }
}

inline void vxt_init_lut64_sel(threadgroup half2* lut64, uint tid, bool is_ec) {
    if (tid < 64u) {
        uint idx4 = tid & 15u;
        ushort2 b = is_ec ? EC_ORTH2D_16_U16[idx4] : ORTH2D_16_U16[idx4];
        ushort s0 = ushort((tid << 11u) & 0x8000u);
        ushort s1 = ushort((tid << 10u) & 0x8000u);
        lut64[tid] = as_type<half2>(ushort2(
            b.x == 0u ? 0u : (b.x | s0),
            b.y == 0u ? 0u : (b.y | s1)
        ));
    }
}

inline void vxt_init_lut64(threadgroup half2* lut64, uint tid) {
    vxt_init_lut64_sel(lut64, tid, false);
}

inline void vxt_init_lut64_4d(threadgroup ushort4* lut64_u16, uint tid) {
    if (tid < 64u) {
        lut64_u16[tid] = ORTH4D_64_U16[tid];
    }
}

inline void vxt_init_lut256_t25(
    threadgroup half2* lut_lo,
    threadgroup half2* lut_hi,
    uint tid,
    uint n_thr
) {
    for (uint idx = tid; idx < 256u; idx += n_thr) {
        ushort4 b_lo = ORTH4D_64_U16[idx >> 2u];
        ushort s0 = ushort((idx & 1u) << 15u);
        ushort s1 = ushort((idx & 2u) << 14u);
        lut_lo[idx] = as_type<half2>(ushort2(
            b_lo.x == 0u ? 0u : (b_lo.x | s0),
            b_lo.y == 0u ? 0u : (b_lo.y | s1)
        ));
        ushort4 b_hi = ORTH4D_64_U16[idx & 63u];
        ushort s2 = ushort((idx << 9u) & 0x8000u);
        ushort s3 = ushort((idx << 8u) & 0x8000u);
        lut_hi[idx] = as_type<half2>(ushort2(
            b_hi.z == 0u ? 0u : (b_hi.z | s2),
            b_hi.w == 0u ? 0u : (b_hi.w | s3)
        ));
    }
}

inline half4 vxt_t25_vec_swz(
    threadgroup const half2* lut_lo,
    threadgroup const half2* lut_hi,
    uint c10_swz
) {
    return half4(lut_lo[c10_swz & 255u], lut_hi[c10_swz >> 2u]);
}

inline uint vxt_t25_unswz(uint swz) {
    return ((swz >> 2u) & 63u) | ((swz & 3u) << 6u) | (swz & 0x300u);
}

inline ushort4 vxt_sgn4_from_c10(uint c10) {
    return ushort4(
        ushort((c10 << 9u) & 0x8000u),
        ushort((c10 << 8u) & 0x8000u),
        ushort((c10 << 7u) & 0x8000u),
        ushort((c10 << 6u) & 0x8000u)
    );
}

inline half4 vxt_orth4d64_const(uint c10) {
    ushort4 base = ORTH4D_64_U16[c10 & 63u];
    ushort4 sgn = vxt_sgn4_from_c10(c10);
    return as_type<half4>(ushort4(
        base.x == 0u ? 0u : (base.x | sgn.x),
        base.y == 0u ? 0u : (base.y | sgn.y),
        base.z == 0u ? 0u : (base.z | sgn.z),
        base.w == 0u ? 0u : (base.w | sgn.w)
    ));
}

inline half4 vxt_t25_vec(threadgroup const ushort4* lut64_u16, uint c10) {
    ushort4 base = lut64_u16[c10 & 63u];
    ushort4 sgn = vxt_sgn4_from_c10(c10);
    return as_type<half4>(ushort4(
        base.x == 0u ? 0u : (base.x | sgn.x),
        base.y == 0u ? 0u : (base.y | sgn.y),
        base.z == 0u ? 0u : (base.z | sgn.z),
        base.w == 0u ? 0u : (base.w | sgn.w)
    ));
}

inline uint2 vxt_t25_unpack_row(device const uint* p80, uint lane, uint sr) {
    uint2 w01 = ((device const uint2*)p80)[lane];
    uint2 raw;
    if (sr == 0u) raw = uint2(w01.x & 1023u, (w01.x >> 10u) & 1023u);
    else if (sr == 1u) raw = uint2((w01.x >> 20u) & 1023u, w01.y & 1023u);
    else if (sr == 2u) raw = uint2((w01.y >> 10u) & 1023u, (w01.y >> 20u) & 1023u);
    else {
        uint h16 = uint(((device const ushort*)(p80 + 64u))[lane]);
        raw = uint2(
            (h16 & 0xffu) | ((w01.x >> 30u) << 8u),
            (h16 >> 8u) | ((w01.y >> 30u) << 8u)
        );
    }
    return uint2(vxt_t25_unswz(raw.x), vxt_t25_unswz(raw.y));
}

inline void vxt_init_lut256_n(threadgroup half4* lut256, uint tid, uint n_thr) {
    for (uint k = tid; k < 256u; k += n_thr) {
        uint c = k >> 4u;
        uint s = k & 15u;
        ushort4 u = ORTH4D_16_U16[c];
        u.x |= ushort((s & 1u) << 15u);
        u.y |= ushort((s & 2u) << 14u);
        u.z |= ushort((s & 4u) << 13u);
        u.w |= ushort((s & 8u) << 12u);
        lut256[(s << 4u) | c] = as_type<half4>(u);
    }
}

inline void vxt_store_tile4(device float* y, uint rows, uint tile, float4 sum4) {
    uint r = tile << 2u;
    if (r + 3u < rows) {
        *(device float4*)(y + r) = sum4;
    } else {
        if (r < rows) y[r] = sum4.x;
        if (r + 1u < rows) y[r + 1u] = sum4.y;
        if (r + 2u < rows) y[r + 2u] = sum4.z;
    }
}

inline void vxt_accum_tile4(device float* y, uint rows, uint tile, float4 sum4, bool add_res) {
    if (!add_res) {
        vxt_store_tile4(y, rows, tile, sum4);
        return;
    }
    uint r = tile << 2u;
    if (r + 3u < rows) {
        *(device float4*)(y + r) += sum4;
    } else {
        if (r < rows) y[r] += sum4.x;
        if (r + 1u < rows) y[r + 1u] += sum4.y;
        if (r + 2u < rows) y[r + 2u] += sum4.z;
    }
}

// Pre-baked 64-entry (256 B) Orth2D-16 signed half2 table in constant cache!
inline half4 vxt_orth4d_const(uint idx, constant ushort4* base16) {
    ushort4 u = base16[idx & 15u];
    u.x |= ushort((idx & 0x10u) << 11u);
    u.y |= ushort((idx & 0x20u) << 10u);
    u.z |= ushort((idx & 0x40u) << 9u);
    u.w |= ushort((idx & 0x80u) << 8u);
    return as_type<half4>(u);
}

inline half2 vxt_orth2d_const(uint idx6, constant ushort2* base16) {
    uint u = as_type<uint>(base16[idx6 & 15u]);
    u |= ((idx6 >> 4u) * 0x40008000u) & 0x80008000u;
    return as_type<half2>(u);
}

inline half2 vxt_orth2d_const_sel(uint idx6, bool is_ec) {
    uint idx4 = idx6 & 15u;
    ushort2 b = is_ec ? EC_ORTH2D_16_U16[idx4] : ORTH2D_16_U16[idx4];
    ushort s0 = ushort((idx6 << 11u) & 0x8000u);
    ushort s1 = ushort((idx6 << 10u) & 0x8000u);
    return as_type<half2>(ushort2(
        b.x == 0u ? 0u : (b.x | s0),
        b.y == 0u ? 0u : (b.y | s1)
    ));
}

inline float4 vxt_eval_turbo2_const(
    uint2 w,
    half4 sh,
    float4 x0,
    float4 x1,
    float4 acc
) {
    uchar4 b01 = as_type<uchar4>(w.x);
    uchar4 b23 = as_type<uchar4>(w.y);
    float d0 = dot(float4(vxt_orth4d_const(b01.x, ORTH4D_16_U16)), x0)
             + dot(float4(vxt_orth4d_const(b01.y, ORTH4D_16_U16)), x1);
    float d1 = dot(float4(vxt_orth4d_const(b01.z, ORTH4D_16_U16)), x0)
             + dot(float4(vxt_orth4d_const(b01.w, ORTH4D_16_U16)), x1);
    float d2 = dot(float4(vxt_orth4d_const(b23.x, ORTH4D_16_U16)), x0)
             + dot(float4(vxt_orth4d_const(b23.y, ORTH4D_16_U16)), x1);
    float d3 = dot(float4(vxt_orth4d_const(b23.z, ORTH4D_16_U16)), x0)
             + dot(float4(vxt_orth4d_const(b23.w, ORTH4D_16_U16)), x1);
    return fma(float4(sh), float4(d0, d1, d2, d3), acc);
}

inline float4 vxt_eval_turbo2_tile(
    uint2 w,
    half4 sh,
    float4 x0,
    float4 x1,
    threadgroup const half4* lut256,
    float4 acc
) {
    uchar4 b01 = as_type<uchar4>(w.x);
    uchar4 b23 = as_type<uchar4>(w.y);
    float d0 = dot(float4(lut256[b01.x]), x0) + dot(float4(lut256[b01.y]), x1);
    float d1 = dot(float4(lut256[b01.z]), x0) + dot(float4(lut256[b01.w]), x1);
    float d2 = dot(float4(lut256[b23.x]), x0) + dot(float4(lut256[b23.y]), x1);
    float d3 = dot(float4(lut256[b23.z]), x0) + dot(float4(lut256[b23.w]), x1);
    return fma(float4(sh), float4(d0, d1, d2, d3), acc);
}

inline float vxt_dot_turbo3_const_sel(uint idx16, uint sgn8, float4 x0, float4 x1, bool is_ec) {
    half2 p0 = vxt_orth2d_const_sel((idx16 & 15u) | ((sgn8 << 4u) & 0x30u), is_ec);
    half2 p1 = vxt_orth2d_const_sel(((idx16 >> 4u) & 15u) | ((sgn8 << 2u) & 0x30u), is_ec);
    half2 p2 = vxt_orth2d_const_sel(((idx16 >> 8u) & 15u) | (sgn8 & 0x30u), is_ec);
    half2 p3 = vxt_orth2d_const_sel((idx16 >> 12u) | ((sgn8 >> 2u) & 0x30u), is_ec);
    return dot(float4(half4(p0, p1)), x0) + dot(float4(half4(p2, p3)), x1);
}

inline float4 vxt_eval_turbo3_const_sel(
    uint2 iw,
    uint sw,
    half4 sh,
    float4 x0,
    float4 x1,
    float4 acc,
    bool is_ec
) {
    ushort4 i4 = as_type<ushort4>(iw);
    uchar4 s4 = as_type<uchar4>(sw);
    float d0 = vxt_dot_turbo3_const_sel(uint(i4.x), uint(s4.x), x0, x1, is_ec);
    float d1 = vxt_dot_turbo3_const_sel(uint(i4.y), uint(s4.y), x0, x1, is_ec);
    float d2 = vxt_dot_turbo3_const_sel(uint(i4.z), uint(s4.z), x0, x1, is_ec);
    float d3 = vxt_dot_turbo3_const_sel(uint(i4.w), uint(s4.w), x0, x1, is_ec);
    return fma(float4(sh), float4(d0, d1, d2, d3), acc);
}

inline float vxt_dot_turbo3_row(
    uint idx16,
    uint sgn8,
    float4 x0,
    float4 x1,
    threadgroup const half2* lut64
) {
    half2 p0 = lut64[(idx16 & 15u) | ((sgn8 << 4u) & 0x30u)];
    half2 p1 = lut64[((idx16 >> 4u) & 15u) | ((sgn8 << 2u) & 0x30u)];
    half2 p2 = lut64[((idx16 >> 8u) & 15u) | (sgn8 & 0x30u)];
    half2 p3 = lut64[(idx16 >> 12u) | ((sgn8 >> 2u) & 0x30u)];
    return dot(float4(half4(p0, p1)), x0) + dot(float4(half4(p2, p3)), x1);
}

inline float4 vxt_eval_turbo3_tile(
    uint2 iw,
    uint sw,
    half4 sh,
    float4 x0,
    float4 x1,
    threadgroup const half2* lut64,
    float4 acc
) {
    ushort4 i4 = as_type<ushort4>(iw);
    uchar4 s4 = as_type<uchar4>(sw);
    float d0 = vxt_dot_turbo3_row(uint(i4.x), uint(s4.x), x0, x1, lut64);
    float d1 = vxt_dot_turbo3_row(uint(i4.y), uint(s4.y), x0, x1, lut64);
    float d2 = vxt_dot_turbo3_row(uint(i4.z), uint(s4.z), x0, x1, lut64);
    float d3 = vxt_dot_turbo3_row(uint(i4.w), uint(s4.w), x0, x1, lut64);
    return fma(float4(sh), float4(d0, d1, d2, d3), acc);
}

// Dual-pipe (Constant Cache + Threadgroup SRAM) on NR=8!
kernel void vxt_gemv_turbo2_nr4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const float* x_rot_pad   [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboParams& p         [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    threadgroup half4 lut256[256];
    vxt_init_lut256(lut256, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;
    uint sg_base = (tg_pos.x * 4u + sg_id) * p.tiles_per_sg;

    for (uint kt = 0u; kt < p.tiles_per_sg; ++kt) {
        uint tile = sg_base + kt;
        if ((tile << 2u) >= p.rows) break;
        device const ushort* sc_tile = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint2* pw_tile = (device const uint2*)(packed_words + ulong(tile) * ulong(steps) * 64ul);

        float4 acc = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
            float4 x0 = xp4[0];
            float4 x1 = xp4[1];
            uint2 w = pw_tile[(step << 5u) + lane];
            half4 sh = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
            acc = vxt_eval_turbo2_tile(w, sh, x0, x1, lut256, acc);
        }

        float4 sum4 = simd_sum(acc);
        if (lane == 0u) {
            vxt_store_tile4(y, p.rows, tile, sum4);
        }
    }
}

kernel void vxt_gemv_turbo2_nr8(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const float* x_rot_pad   [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboParams& p         [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    threadgroup half4 lut256[256];
    vxt_init_lut256(lut256, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;
    uint sg_pair_base = (tg_pos.x * 4u + sg_id) * p.tiles_per_sg;

    for (uint kp = 0u; kp < p.tiles_per_sg; ++kp) {
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
                float4 x0 = xp4[0];
                float4 x1 = xp4[1];
                uint off = (step << 5u) + lane;
                uint soff = (step << 3u) + (gh << 2u);
                acc0 = vxt_eval_turbo2_tile(pw0[off], *(device const half4*)(sc0 + soff), x0, x1, lut256, acc0);
                acc1 = vxt_eval_turbo2_const(pw1[off], *(device const half4*)(sc1 + soff), x0, x1, acc1);
            }
        } else {
            for (uint step = 0u; step < steps; ++step) {
                device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
                float4 x0 = xp4[0];
                float4 x1 = xp4[1];
                acc0 = vxt_eval_turbo2_tile(
                    pw0[(step << 5u) + lane],
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)),
                    x0, x1, lut256, acc0
                );
            }
        }

        float4 sum0 = simd_sum(acc0);
        float4 sum1 = simd_sum(acc1);
        if (lane == 0u) {
            vxt_store_tile4(y, p.rows, tile0, sum0);
            if (has_t1) {
                vxt_store_tile4(y, p.rows, tile0 + 1u, sum1);
            }
        }
    }
}

kernel void vxt_gemv_turbo3_nr4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const float* x_rot_pad   [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboParams& p         [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool is_ec = (p.tiles_per_sg & 256u) != 0u;
    uint tps = p.tiles_per_sg & 255u;
    threadgroup half2 lut64[64];
    vxt_init_lut64_sel(lut64, tid, is_ec);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;
    uint sg_base = (tg_pos.x * 2u + sg_id) * tps;

    for (uint kt = 0u; kt < tps; ++kt) {
        uint tile = sg_base + kt;
        if ((tile << 2u) >= p.rows) break;
        device const ushort* sc_tile = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw_tile = packed_words + ulong(tile) * ulong(steps) * 96ul;

        float4 acc = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
            float4 x0 = xp4[0];
            float4 x1 = xp4[1];
            device const uint* p96 = pw_tile + step * 96u;
            uint2 iw = ((device const uint2*)p96)[lane];
            uint sw = p96[64u + lane];
            half4 sh = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
            acc = vxt_eval_turbo3_tile(iw, sw, sh, x0, x1, lut64, acc);
        }

        float4 sum4 = simd_sum(acc);
        if (lane == 0u) {
            vxt_store_tile4(y, p.rows, tile, sum4);
        }
    }
}

kernel void vxt_gemv_turbo3_nr8(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const float* x_rot_pad   [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboParams& p         [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    bool is_ec = (p.tiles_per_sg & 256u) != 0u;
    uint tps = p.tiles_per_sg & 255u;
    threadgroup half2 lut64[64];
    vxt_init_lut64_sel(lut64, tid, is_ec);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;
    uint sg_pair_base = (tg_pos.x * 2u + sg_id) * tps;

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
                float4 x0 = xp4[0];
                float4 x1 = xp4[1];
                device const uint* p0 = pw0 + step * 96u;
                device const uint* p1 = pw1 + step * 96u;
                uint soff = (step << 3u) + (gh << 2u);
                acc0 = vxt_eval_turbo3_tile(
                    ((device const uint2*)p0)[lane], p0[64u + lane],
                    *(device const half4*)(sc0 + soff), x0, x1, lut64, acc0
                );
                acc1 = vxt_eval_turbo3_const_sel(
                    ((device const uint2*)p1)[lane], p1[64u + lane],
                    *(device const half4*)(sc1 + soff), x0, x1, acc1, is_ec
                );
            }
        } else {
            for (uint step = 0u; step < steps; ++step) {
                device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
                float4 x0 = xp4[0];
                float4 x1 = xp4[1];
                device const uint* p0 = pw0 + step * 96u;
                acc0 = vxt_eval_turbo3_tile(
                    ((device const uint2*)p0)[lane], p0[64u + lane],
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), x0, x1, lut64, acc0
                );
            }
        }

        float4 sum0 = simd_sum(acc0);
        float4 sum1 = simd_sum(acc1);
        if (lane == 0u) {
            vxt_store_tile4(y, p.rows, tile0, sum0);
            if (has_t1) {
                vxt_store_tile4(y, p.rows, tile0 + 1u, sum1);
            }
        }
    }
}

inline float4 vxt_eval_turbo25_tile(
    uint2 w01,
    uint h16,
    half4 sh,
    float4 x0,
    float4 x1,
    threadgroup const ushort4* lut64_u16,
    float4 acc
) {
    uint c00 = w01.x & 1023u, c01 = (w01.x >> 10u) & 1023u;
    uint c10 = (w01.x >> 20u) & 1023u, c11 = w01.y & 1023u;
    uint c20 = (w01.y >> 10u) & 1023u, c21 = (w01.y >> 20u) & 1023u;
    uint c30 = (h16 & 0xffu) | ((w01.x >> 30u) << 8u);
    uint c31 = (h16 >> 8u) | ((w01.y >> 30u) << 8u);
    float d0 = dot(float4(vxt_t25_vec(lut64_u16, c00)), x0) + dot(float4(vxt_t25_vec(lut64_u16, c01)), x1);
    float d1 = dot(float4(vxt_t25_vec(lut64_u16, c10)), x0) + dot(float4(vxt_t25_vec(lut64_u16, c11)), x1);
    float d2 = dot(float4(vxt_t25_vec(lut64_u16, c20)), x0) + dot(float4(vxt_t25_vec(lut64_u16, c21)), x1);
    float d3 = dot(float4(vxt_t25_vec(lut64_u16, c30)), x0) + dot(float4(vxt_t25_vec(lut64_u16, c31)), x1);
    return fma(float4(sh), float4(d0, d1, d2, d3), acc);
}

kernel void vxt_gemv_turbo25_nr4(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const float* x_rot_pad   [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboParams& p         [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    threadgroup ushort4 lut64_u16[64];
    vxt_init_lut64_4d(lut64_u16, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;
    uint sg_base = (tg_pos.x * 4u + sg_id) * p.tiles_per_sg;

    for (uint kt = 0u; kt < p.tiles_per_sg; ++kt) {
        uint tile = sg_base + kt;
        if ((tile << 2u) >= p.rows) break;
        device const ushort* sc_tile = scales_f16 + ulong(tile) * ulong(steps) * 8ul;
        device const uint* pw_tile = packed_words + ulong(tile) * ulong(steps) * 80ul;

        float4 acc = float4(0.0f);
        for (uint step = 0u; step < steps; ++step) {
            device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
            device const uint* p80 = pw_tile + step * 80u;
            uint2 w01 = ((device const uint2*)p80)[lane];
            uint h16 = uint(((device const ushort*)(p80 + 64u))[lane]);
            half4 sh = *(device const half4*)(sc_tile + (step << 3u) + (gh << 2u));
            acc = vxt_eval_turbo25_tile(w01, h16, sh, xp4[0], xp4[1], lut64_u16, acc);
        }

        float4 sum4 = simd_sum(acc);
        if (lane == 0u) {
            vxt_store_tile4(y, p.rows, tile, sum4);
        }
    }
}

kernel void vxt_gemv_turbo25_nr8(
    device const ushort* scales_f16 [[buffer(0)]],
    device const uint* packed_words [[buffer(1)]],
    device const float* x_rot_pad   [[buffer(2)]],
    device float* y                 [[buffer(3)]],
    constant TurboParams& p         [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint sg_id                      [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]],
    uint3 tg_pos                    [[threadgroup_position_in_grid]]
) {
    threadgroup ushort4 lut64_u16[64];
    vxt_init_lut64_4d(lut64_u16, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint steps = p.padded_cols >> 8u;
    uint gh = lane >> 4u;
    uint sg_pair_base = (tg_pos.x * 4u + sg_id) * p.tiles_per_sg;

    for (uint kp = 0u; kp < p.tiles_per_sg; ++kp) {
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
                device const uint* p0 = pw0 + step * 80u;
                device const uint* p1 = pw1 + step * 80u;
                uint soff = (step << 3u) + (gh << 2u);
                acc0 = vxt_eval_turbo25_tile(
                    ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                    *(device const half4*)(sc0 + soff), xp4[0], xp4[1], lut64_u16, acc0
                );
                acc1 = vxt_eval_turbo25_tile(
                    ((device const uint2*)p1)[lane], uint(((device const ushort*)(p1 + 64u))[lane]),
                    *(device const half4*)(sc1 + soff), xp4[0], xp4[1], lut64_u16, acc1
                );
            }
        } else {
            for (uint step = 0u; step < steps; ++step) {
                device const float4* xp4 = (device const float4*)(x_rot_pad + (step << 8u) + (lane << 3u));
                device const uint* p0 = pw0 + step * 80u;
                acc0 = vxt_eval_turbo25_tile(
                    ((device const uint2*)p0)[lane], uint(((device const ushort*)(p0 + 64u))[lane]),
                    *(device const half4*)(sc0 + (step << 3u) + (gh << 2u)), xp4[0], xp4[1], lut64_u16, acc0
                );
            }
        }

        float4 sum0 = simd_sum(acc0);
        float4 sum1 = simd_sum(acc1);
        if (lane == 0u) {
            vxt_store_tile4(y, p.rows, tile0, sum0);
            if (has_t1) vxt_store_tile4(y, p.rows, tile0 + 1u, sum1);
        }
    }
}
