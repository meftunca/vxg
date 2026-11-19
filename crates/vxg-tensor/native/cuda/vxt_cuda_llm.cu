// Native NVIDIA CUDA 100% VRAM-resident VXT-Turbo / VXT-Turbo-HQ LLM Runner.
// Supports hybrid Gated-DeltaNet + Gated-GQA (Qwen3.5) and dense GQA Transformers
// (Llama / Qwen2 / Qwen3 / Gemma) with zero PCIe copies between layers.
// Compiled when VXG_CUDA=1 on Linux/NVIDIA.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

namespace {

constexpr int MAT_NONE = 0;
constexpr int MAT_F32 = 1;
constexpr int MAT_E8 = 7;
constexpr int MAT_E8_HQ = 8;

constexpr float INV_SQRT32 = 0.1767766952966369f;

struct DevMat {
    int kind = MAT_NONE;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t padded_cols = 0;
    uint16_t* d_scales = nullptr;
    uint32_t* d_words = nullptr;
    float* d_f32 = nullptr;

    void free_dev() {
        if (d_scales) cudaFree(d_scales);
        if (d_words) cudaFree(d_words);
        if (d_f32) cudaFree(d_f32);
        d_scales = nullptr;
        d_words = nullptr;
        d_f32 = nullptr;
        kind = MAT_NONE;
    }
};

struct DevLayer {
    bool is_deltanet = false;
    bool gated_q = false;
    float* d_attn_norm = nullptr;
    float* d_post_attn_norm = nullptr;
    float* d_attn_post_norm = nullptr;
    float* d_ffn_post_norm = nullptr;

    // DeltaNet weights
    DevMat wqkv;
    DevMat wqkv_gate;
    DevMat ssm_alpha;
    DevMat ssm_beta;
    DevMat ssm_out;
    float* d_ssm_conv1d = nullptr;
    float* d_ssm_a = nullptr;
    float* d_ssm_dt_bias = nullptr;
    float* d_ssm_norm = nullptr;

    // Full GQA weights
    DevMat wq;
    DevMat wk;
    DevMat wv;
    DevMat wo;
    float* d_q_norm = nullptr;
    float* d_k_norm = nullptr;
    float* d_q_bias = nullptr;
    float* d_k_bias = nullptr;
    float* d_v_bias = nullptr;

    // FFN weights
    DevMat ffn_gate;
    DevMat ffn_up;
    DevMat ffn_down;

    void free_dev() {
        auto ff = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
        ff(d_attn_norm); ff(d_post_attn_norm); ff(d_attn_post_norm); ff(d_ffn_post_norm);
        wqkv.free_dev(); wqkv_gate.free_dev(); ssm_alpha.free_dev(); ssm_beta.free_dev(); ssm_out.free_dev();
        ff(d_ssm_conv1d); ff(d_ssm_a); ff(d_ssm_dt_bias); ff(d_ssm_norm);
        wq.free_dev(); wk.free_dev(); wv.free_dev(); wo.free_dev();
        ff(d_q_norm); ff(d_k_norm); ff(d_q_bias); ff(d_k_bias); ff(d_v_bias);
        ffn_gate.free_dev(); ffn_up.free_dev(); ffn_down.free_dev();
    }
};

struct CudaLlmRunner {
    uint32_t n_vocab = 0;
    uint32_t n_embd = 0;
    uint32_t n_layer = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_ff = 0;
    uint32_t rope_dim = 0;
    float rope_freq_base = 10000.0f;
    bool rope_neox = false;
    float rms_norm_eps = 1e-6f;
    float embed_scale = 1.0f;
    bool ffn_gelu = false;

    uint32_t ssm_conv_kernel = 0;
    uint32_t ssm_state_size = 0;
    uint32_t ssm_group_count = 0;
    uint32_t ssm_dt_rank = 0;
    uint32_t ssm_inner_size = 0;

    uint32_t padded_embd = 0;
    uint32_t padded_ff = 0;
    uint32_t max_padded_cols = 0;
    uint32_t conv_ch = 0;
    uint32_t dv = 0;

    DevMat token_embd;
    DevMat output_head;  // kind == MAT_NONE means tied to token_embd
    float* d_output_norm = nullptr;
    std::vector<DevLayer> layers;

    ~CudaLlmRunner() {
        token_embd.free_dev();
        output_head.free_dev();
        if (d_output_norm) cudaFree(d_output_norm);
        for (auto& ly : layers) ly.free_dev();
    }
};

struct CudaLlmState {
    const CudaLlmRunner* runner = nullptr;
    cudaStream_t stream = nullptr;

    // Scratch activation buffers (all VRAM)
    float* d_cur = nullptr;
    float* d_normed = nullptr;
    float* d_xrot = nullptr;
    float* d_tmp = nullptr;

    // DeltaNet scratch
    float* d_qkv = nullptr;
    float* d_z = nullptr;
    float* d_alpha = nullptr;
    float* d_beta = nullptr;
    float* d_conv_out = nullptr;
    float* d_q_l2 = nullptr;
    float* d_k_l2 = nullptr;
    float* d_o_buf = nullptr;

    // GQA scratch
    float* d_qraw = nullptr;
    float* d_q = nullptr;
    float* d_qgate = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;

    // FFN & logits scratch
    float* d_gate = nullptr;
    float* d_up = nullptr;
    float* d_logits = nullptr;
    uint8_t* d_seen = nullptr;
    float* d_arg_val = nullptr;
    uint32_t* d_arg_idx = nullptr;
    uint32_t* d_next_tok = nullptr;
    uint32_t* h_next_tok = nullptr;  // pinned 4-byte host readback slot

    // Persistent per-layer states
    std::vector<float*> d_conv_states;
    std::vector<float*> d_ssm_states;
    std::vector<float*> d_k_caches;
    std::vector<float*> d_v_caches;
    uint32_t kv_cap = 0;

    ~CudaLlmState() {
        auto ff = [](void* p) { if (p) cudaFree(p); };
        ff(d_cur); ff(d_normed); ff(d_xrot); ff(d_tmp);
        ff(d_qkv); ff(d_z); ff(d_alpha); ff(d_beta); ff(d_conv_out); ff(d_q_l2); ff(d_k_l2); ff(d_o_buf);
        ff(d_qraw); ff(d_q); ff(d_qgate); ff(d_k); ff(d_v);
        ff(d_gate); ff(d_up); ff(d_logits); ff(d_seen); ff(d_arg_val); ff(d_arg_idx); ff(d_next_tok);
        if (h_next_tok) cudaFreeHost(h_next_tok);
        for (auto* p : d_conv_states) ff(p);
        for (auto* p : d_ssm_states) ff(p);
        for (auto* p : d_k_caches) ff(p);
        for (auto* p : d_v_caches) ff(p);
        if (stream) cudaStreamDestroy(stream);
    }
};

// ---------------- Device Helpers ----------------

__device__ __forceinline__ uint32_t col_sign_mask(uint32_t col) {
    uint32_t h = col ^ 0x9e3779b9u;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return (h & 1u) << 31u;
}

__device__ __forceinline__ float apply_col_sign(float v, uint32_t col) {
    if (v == 0.0f) return 0.0f;
    return __uint_as_float(__float_as_uint(v) ^ col_sign_mask(col));
}

__device__ __forceinline__ float warp_fwht32(float v, uint32_t lane) {
    for (uint32_t mask = 1u; mask < 32u; mask <<= 1u) {
        float other = __shfl_xor_sync(0xffffffffu, v, mask);
        v = ((lane & mask) == 0u) ? (v + other) : (other - v);
    }
    return v * INV_SQRT32;
}

__device__ __forceinline__ float warp_sum_all(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__device__ __forceinline__ float4 warp_sum4_down(float4 v) {
    for (int off = 16; off > 0; off >>= 1) {
        v.x += __shfl_down_sync(0xffffffffu, v.x, off);
        v.y += __shfl_down_sync(0xffffffffu, v.y, off);
        v.z += __shfl_down_sync(0xffffffffu, v.z, off);
        v.w += __shfl_down_sync(0xffffffffu, v.w, off);
    }
    return v;
}

__device__ __forceinline__ float4 load_half4(const uint16_t* p) {
    const __half* hp = reinterpret_cast<const __half*>(p);
    return make_float4(__half2float(hp[0]), __half2float(hp[1]), __half2float(hp[2]), __half2float(hp[3]));
}

__device__ __forceinline__ float dot_e8(uint32_t rw, float4 x0, float4 x1) {
    uint32_t m7 = rw & 0x0fffffffu;
    uint32_t par = m7 ^ (m7 >> 4u);
    par ^= par >> 8u;
    uint32_t u7 = ((rw >> 27u) & 14u) | ((par ^ (par >> 16u)) & 1u);
    float bias = (rw & 0x80000000u) ? -7.5f : -8.0f;

    float s0 = fmaf(float(rw & 0xfu) + bias, x0.x,
               fmaf(float((rw >> 4u) & 0xfu) + bias, x0.y,
               fmaf(float((rw >> 8u) & 0xfu) + bias, x0.z,
                    (float((rw >> 12u) & 0xfu) + bias) * x0.w)));
    float s1 = fmaf(float((rw >> 16u) & 0xfu) + bias, x1.x,
               fmaf(float((rw >> 20u) & 0xfu) + bias, x1.y,
               fmaf(float((rw >> 24u) & 0xfu) + bias, x1.z,
                    (float(u7) + bias) * x1.w)));
    return s0 + s1;
}

__device__ __forceinline__ float dot_e8_hq(uint32_t w0, uint32_t w1, float4 x0, float4 x1) {
    uint32_t b0 = w0 & 0xffu;
    uint32_t b1 = (w0 >> 8u) & 0xffu;
    uint32_t b2 = (w0 >> 16u) & 0xffu;
    uint32_t b3 = (w0 >> 24u) & 0xffu;
    uint32_t b4 = w1 & 0xffu;
    uint32_t b5 = (w1 >> 8u) & 0xffu;
    uint32_t b6 = (w1 >> 16u) & 0xffu;
    uint32_t tb = (w1 >> 24u) & 0xffu;
    uint32_t par = (b0 ^ b1 ^ b2 ^ b3 ^ b4 ^ b5 ^ b6) & 1u;
    uint32_t u7 = ((tb & 0x7fu) << 1u) | par;
    float bias = (tb & 0x80u) ? -127.5f : -128.0f;

    float s0 = fmaf(float(b0) + bias, x0.x,
               fmaf(float(b1) + bias, x0.y,
               fmaf(float(b2) + bias, x0.z, (float(b3) + bias) * x0.w)));
    float s1 = fmaf(float(b4) + bias, x1.x,
               fmaf(float(b5) + bias, x1.y,
               fmaf(float(b6) + bias, x1.z, (float(u7) + bias) * x1.w)));
    return s0 + s1;
}

template <int BITS>
__device__ __forceinline__ void accum_tile_step(
    const uint16_t* sc_tile,
    const uint32_t* pw_tile,
    uint32_t step,
    uint32_t lane,
    float4 x0,
    float4 x1,
    float4& acc
) {
    uint32_t b8 = lane >> 2u;
    float4 sh = load_half4(sc_tile + (step << 5u) + (b8 << 2u));
    if constexpr (BITS == MAT_E8) {
        uint4 w = reinterpret_cast<const uint4*>(pw_tile + step * 128u)[lane];
        acc.x = fmaf(sh.x, dot_e8(w.x, x0, x1), acc.x);
        acc.y = fmaf(sh.y, dot_e8(w.y, x0, x1), acc.y);
        acc.z = fmaf(sh.z, dot_e8(w.z, x0, x1), acc.z);
        acc.w = fmaf(sh.w, dot_e8(w.w, x0, x1), acc.w);
    } else {
        const uint4* p = reinterpret_cast<const uint4*>(pw_tile + step * 256u);
        uint4 wa = p[lane * 2u], wb = p[lane * 2u + 1u];
        acc.x = fmaf(sh.x, dot_e8_hq(wa.x, wa.y, x0, x1), acc.x);
        acc.y = fmaf(sh.y, dot_e8_hq(wa.z, wa.w, x0, x1), acc.y);
        acc.z = fmaf(sh.z, dot_e8_hq(wb.x, wb.y, x0, x1), acc.z);
        acc.w = fmaf(sh.w, dot_e8_hq(wb.z, wb.w, x0, x1), acc.w);
    }
}

__device__ __forceinline__ void store_tile4(float* y, uint32_t rows, uint32_t tile, float4 s, bool add_res, bool atomic_add) {
    uint32_t r = tile << 2u;
    if (atomic_add) {
        if (r < rows) atomicAdd(&y[r], s.x);
        if (r + 1u < rows) atomicAdd(&y[r + 1u], s.y);
        if (r + 2u < rows) atomicAdd(&y[r + 2u], s.z);
        if (r + 3u < rows) atomicAdd(&y[r + 3u], s.w);
    } else if (add_res) {
        if (r + 3u < rows) {
            float4 cur = reinterpret_cast<const float4*>(y + r)[0];
            cur.x += s.x; cur.y += s.y; cur.z += s.z; cur.w += s.w;
            reinterpret_cast<float4*>(y + r)[0] = cur;
        } else {
            if (r < rows) y[r] += s.x;
            if (r + 1u < rows) y[r + 1u] += s.y;
            if (r + 2u < rows) y[r + 2u] += s.z;
        }
    } else {
        if (r + 3u < rows) {
            reinterpret_cast<float4*>(y + r)[0] = s;
        } else {
            if (r < rows) y[r] = s.x;
            if (r + 1u < rows) y[r + 1u] = s.y;
            if (r + 2u < rows) y[r + 2u] = s.z;
        }
    }
}

// ---------------- Embedding Kernels ----------------

template <int BITS>
__global__ void k_embed_turbo(
    const uint16_t* __restrict__ scales_f16,
    const uint32_t* __restrict__ packed_words,
    uint32_t token_id,
    uint32_t cols,
    uint32_t padded_cols,
    float embed_scale,
    float* __restrict__ out_cur
) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= padded_cols) return;
    uint32_t lane = threadIdx.x & 31u;
    uint32_t tile = token_id >> 2u;
    uint32_t sr = token_id & 3u;
    uint32_t steps = padded_cols >> 8u;
    uint32_t step = col >> 8u;
    uint32_t in_step = col & 255u;
    uint32_t lane8 = in_step >> 3u;
    uint32_t k = in_step & 7u;
    uint32_t b8 = lane8 >> 2u;

    size_t tile_step = static_cast<size_t>(tile) * steps + step;
    float sc = __half2float(__ushort_as_half(scales_f16[tile_step * 32u + b8 * 4u + sr]));
    float qv = 0.0f;
    if constexpr (BITS == MAT_E8) {
        uint32_t rw = packed_words[tile_step * 128u + lane8 * 4u + sr];
        uint32_t m7 = rw & 0x0fffffffu;
        uint32_t par = m7 ^ (m7 >> 4u);
        par ^= par >> 8u;
        uint32_t u7 = ((rw >> 27u) & 14u) | ((par ^ (par >> 16u)) & 1u);
        float bias = (rw & 0x80000000u) ? -7.5f : -8.0f;
        uint32_t u = (k < 7u) ? ((rw >> (k * 4u)) & 0xfu) : u7;
        qv = float(u) + bias;
    } else {
        size_t idx = tile_step * 256u + static_cast<size_t>(lane8 * 4u + sr) * 2u;
        uint32_t w0 = packed_words[idx];
        uint32_t w1 = packed_words[idx + 1u];
        uint32_t b0 = w0 & 0xffu, b1 = (w0 >> 8u) & 0xffu, b2 = (w0 >> 16u) & 0xffu, b3 = (w0 >> 24u) & 0xffu;
        uint32_t b4 = w1 & 0xffu, b5 = (w1 >> 8u) & 0xffu, b6 = (w1 >> 16u) & 0xffu, tb = (w1 >> 24u) & 0xffu;
        uint32_t par = (b0 ^ b1 ^ b2 ^ b3 ^ b4 ^ b5 ^ b6) & 1u;
        uint32_t u7 = ((tb & 0x7fu) << 1u) | par;
        float bias = (tb & 0x80u) ? -127.5f : -128.0f;
        uint32_t u = (k < 4u) ? ((w0 >> (k * 8u)) & 0xffu) : ((k < 7u) ? ((w1 >> ((k - 4u) * 8u)) & 0xffu) : u7);
        qv = float(u) + bias;
    }
    float v = warp_fwht32(sc * qv, lane);
    if (col < cols) {
        out_cur[col] = apply_col_sign(v, col) * embed_scale;
    }
}

__global__ void k_embed_f32(
    const float* __restrict__ table,
    uint32_t token_id,
    uint32_t cols,
    float embed_scale,
    float* __restrict__ out_cur
) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        out_cur[col] = table[static_cast<size_t>(token_id) * cols + col] * embed_scale;
    }
}

// ---------------- Fused Norm & Activation + FWHT-32 Kernels ----------------

// Single block (256 threads = 8 warps): RMSNorm -> optional normed_out + fused col_sign & FWHT-32 -> xrot_out
__global__ void k_rmsnorm_fwht32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float eps,
    float* __restrict__ normed_out,
    float* __restrict__ xrot_out,
    uint32_t dim,
    uint32_t padded_dim
) {
    __shared__ float s_warp[8];
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid >> 5u;
    uint32_t lane = tid & 31u;

    float sq = 0.0f;
    for (uint32_t i = tid; i < dim; i += 256u) {
        float v = x[i];
        sq = fmaf(v, v, sq);
    }
    sq = warp_sum_all(sq);
    if (lane == 0u) s_warp[wid] = sq;
    __syncthreads();
    if (wid == 0u) {
        float total = (lane < 8u) ? s_warp[lane] : 0.0f;
        total = warp_sum_all(total);
        if (lane == 0u) s_warp[0] = rsqrtf(total / float(dim) + eps);
    }
    __syncthreads();
    float inv_rms = s_warp[0];

    for (uint32_t base = wid * 32u; base < padded_dim; base += 256u) {
        uint32_t idx = base + lane;
        float nv = (idx < dim) ? (x[idx] * inv_rms * w[idx]) : 0.0f;
        if (normed_out && idx < dim) normed_out[idx] = nv;
        float xr = warp_fwht32((idx < dim) ? apply_col_sign(nv, idx) : 0.0f, lane);
        if (idx < padded_dim) xrot_out[idx] = xr;
    }
}

// Optional Gemma post-norm + residual add: cur[i] += rms_norm(tmp, w)[i]
__global__ void k_rmsnorm_add(
    const float* __restrict__ tmp,
    const float* __restrict__ w,
    float eps,
    float* __restrict__ cur,
    uint32_t dim
) {
    __shared__ float s_warp[8];
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid >> 5u;
    uint32_t lane = tid & 31u;
    float sq = 0.0f;
    for (uint32_t i = tid; i < dim; i += 256u) {
        float v = tmp[i];
        sq = fmaf(v, v, sq);
    }
    sq = warp_sum_all(sq);
    if (lane == 0u) s_warp[wid] = sq;
    __syncthreads();
    if (wid == 0u) {
        float total = (lane < 8u) ? s_warp[lane] : 0.0f;
        total = warp_sum_all(total);
        if (lane == 0u) s_warp[0] = rsqrtf(total / float(dim) + eps);
    }
    __syncthreads();
    float inv_rms = s_warp[0];
    for (uint32_t i = tid; i < dim; i += 256u) {
        cur[i] += tmp[i] * inv_rms * w[i];
    }
}

__global__ void k_swiglu_fwht32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ xrot_out,
    uint32_t n_ff,
    uint32_t padded_ff,
    int use_gelu
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_ff) return;
    uint32_t lane = threadIdx.x & 31u;
    float v = 0.0f;
    if (idx < n_ff) {
        float g = gate[idx];
        float u = up[idx];
        float act;
        if (!use_gelu) {
            act = (g / (1.0f + expf(-g))) * u;
        } else {
            float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
            act = 0.5f * g * (1.0f + tanhf(inner)) * u;
        }
        v = apply_col_sign(act, idx);
    }
    xrot_out[idx] = warp_fwht32(v, lane);
}

// ---------------- High-Occupancy GEMV Kernels ----------------

// 4 warps/block, 2 tiles (8 rows)/warp — optimal for large matrices (LM-Head, wide layers).
template <int BITS>
__global__ void k_gemv_turbo_nr8(
    const uint16_t* __restrict__ scales_f16,
    const uint32_t* __restrict__ packed_words,
    const float* __restrict__ x_rot,
    float* __restrict__ y_out,
    uint32_t rows,
    uint32_t padded_cols
) {
    uint32_t warp_id = (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    uint32_t tile0 = warp_id * 2u;
    if ((tile0 << 2u) >= rows) return;
    bool has_t1 = ((tile0 + 1u) << 2u) < rows;
    uint32_t steps = padded_cols >> 8u;
    size_t sc_stride = static_cast<size_t>(steps) * 32u;
    size_t pw_stride = static_cast<size_t>(steps) * ((BITS == MAT_E8) ? 128u : 256u);

    const uint16_t* sc0 = scales_f16 + static_cast<size_t>(tile0) * sc_stride;
    const uint32_t* pw0 = packed_words + static_cast<size_t>(tile0) * pw_stride;
    float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (has_t1) {
        const uint16_t* sc1 = sc0 + sc_stride;
        const uint32_t* pw1 = pw0 + pw_stride;
        for (uint32_t step = 0; step < steps; ++step) {
            const float4* xp = reinterpret_cast<const float4*>(x_rot + (step << 8u) + (lane << 3u));
            float4 x0 = xp[0], x1 = xp[1];
            accum_tile_step<BITS>(sc0, pw0, step, lane, x0, x1, acc0);
            accum_tile_step<BITS>(sc1, pw1, step, lane, x0, x1, acc1);
        }
    } else {
        for (uint32_t step = 0; step < steps; ++step) {
            const float4* xp = reinterpret_cast<const float4*>(x_rot + (step << 8u) + (lane << 3u));
            float4 x0 = xp[0], x1 = xp[1];
            accum_tile_step<BITS>(sc0, pw0, step, lane, x0, x1, acc0);
        }
    }

    float4 s0 = warp_sum4_down(acc0);
    float4 s1 = warp_sum4_down(acc1);
    if (lane == 0u) {
        store_tile4(y_out, rows, tile0, s0, false, false);
        if (has_t1) store_tile4(y_out, rows, tile0 + 1u, s1, false, false);
    }
}

// Split-K 1 tile (4 rows)/warp — saturates all 170 SMs on small-dim residual projections!
template <int BITS, bool ADD_RES>
__global__ void k_gemv_turbo_sk(
    const uint16_t* __restrict__ scales_f16,
    const uint32_t* __restrict__ packed_words,
    const float* __restrict__ x_rot,
    float* __restrict__ y_out,
    uint32_t rows,
    uint32_t padded_cols,
    uint32_t split_k
) {
    uint32_t tile = (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
    uint32_t split = blockIdx.y;
    uint32_t lane = threadIdx.x & 31u;
    if ((tile << 2u) >= rows) return;

    uint32_t steps = padded_cols >> 8u;
    uint32_t step_beg = (steps * split) / split_k;
    uint32_t step_end = (steps * (split + 1u)) / split_k;
    if (step_beg >= step_end) return;

    size_t sc_stride = static_cast<size_t>(steps) * 32u;
    size_t pw_stride = static_cast<size_t>(steps) * ((BITS == MAT_E8) ? 128u : 256u);
    const uint16_t* sc = scales_f16 + static_cast<size_t>(tile) * sc_stride;
    const uint32_t* pw = packed_words + static_cast<size_t>(tile) * pw_stride;

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t step = step_beg; step < step_end; ++step) {
        const float4* xp = reinterpret_cast<const float4*>(x_rot + (step << 8u) + (lane << 3u));
        accum_tile_step<BITS>(sc, pw, step, lane, xp[0], xp[1], acc);
    }
    float4 s = warp_sum4_down(acc);
    if (lane == 0u) {
        store_tile4(y_out, rows, tile, s, ADD_RES, split_k > 1u);
    }
}

struct MultiTurboDesc {
    const uint16_t* sc;
    const uint32_t* pw;
    float* y;
    uint32_t rows;
    uint32_t tile_off;
    uint32_t tiles;
};

// Fused 2-matrix or 3-matrix GEMV sharing x_rot in 1 kernel launch!
template <int BITS, int N_MATS>
__global__ void k_gemv_turbo_multi(
    MultiTurboDesc m0,
    MultiTurboDesc m1,
    MultiTurboDesc m2,
    const float* __restrict__ x_rot,
    uint32_t padded_cols
) {
    uint32_t gtile = (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;

    MultiTurboDesc m;
    uint32_t tile;
    if (gtile < m0.tiles) {
        m = m0; tile = gtile;
    } else if (gtile < m1.tile_off + m1.tiles) {
        m = m1; tile = gtile - m1.tile_off;
    } else if constexpr (N_MATS == 3) {
        if (gtile < m2.tile_off + m2.tiles) {
            m = m2; tile = gtile - m2.tile_off;
        } else {
            return;
        }
    } else {
        return;
    }

    uint32_t steps = padded_cols >> 8u;
    size_t sc_stride = static_cast<size_t>(steps) * 32u;
    size_t pw_stride = static_cast<size_t>(steps) * ((BITS == MAT_E8) ? 128u : 256u);
    const uint16_t* sc = m.sc + static_cast<size_t>(tile) * sc_stride;
    const uint32_t* pw = m.pw + static_cast<size_t>(tile) * pw_stride;

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t step = 0; step < steps; ++step) {
        const float4* xp = reinterpret_cast<const float4*>(x_rot + (step << 8u) + (lane << 3u));
        accum_tile_step<BITS>(sc, pw, step, lane, xp[0], xp[1], acc);
    }
    float4 s = warp_sum4_down(acc);
    if (lane == 0u) {
        store_tile4(m.y, m.rows, tile, s, false, false);
    }
}

template <bool ADD_RES>
__global__ void k_gemv_f32_single(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t rows,
    uint32_t cols
) {
    uint32_t row = (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= rows) return;
    const float* wr = w + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (uint32_t c = lane; c < cols; c += 32u) {
        sum = fmaf(wr[c], x[c], sum);
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    }
    if (lane == 0u) {
        if constexpr (ADD_RES) y[row] += sum;
        else y[row] = sum;
    }
}

// Fused ssm_alpha + ssm_beta f32 GEMV (1 block per row, 128 threads/row for fast reduction on tiny row counts).
__global__ void k_gemv_f32_pair(
    const float* __restrict__ w0,
    float* __restrict__ y0,
    uint32_t rows0,
    const float* __restrict__ w1,
    float* __restrict__ y1,
    uint32_t rows1,
    const float* __restrict__ x,
    uint32_t cols
) {
    __shared__ float s_w[4];
    uint32_t grow = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid >> 5u;
    uint32_t lane = tid & 31u;

    const float* wr;
    float* dst;
    if (grow < rows0) {
        wr = w0 + static_cast<size_t>(grow) * cols;
        dst = y0 + grow;
    } else if (grow < rows0 + rows1) {
        uint32_t r = grow - rows0;
        wr = w1 + static_cast<size_t>(r) * cols;
        dst = y1 + r;
    } else {
        return;
    }
    float sum = 0.0f;
    for (uint32_t c = tid; c < cols; c += 128u) {
        sum = fmaf(wr[c], x[c], sum);
    }
    sum = warp_sum_all(sum);
    if (lane == 0u) s_w[wid] = sum;
    __syncthreads();
    if (wid == 0u) {
        float tot = (lane < 4u) ? s_w[lane] : 0.0f;
        tot = warp_sum_all(tot);
        if (lane == 0u) *dst = tot;
    }
}

// ---------------- Gated-DeltaNet Kernels ----------------

__global__ void k_deltanet_conv_silu(
    const float* __restrict__ qkv_mixed,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_state,
    float* __restrict__ conv_silu,
    uint32_t conv_ch,
    uint32_t k_sz
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_ch) return;
    uint32_t hist = (k_sz > 0u) ? (k_sz - 1u) : 0u;
    const float* w = conv_w + static_cast<size_t>(c) * k_sz;
    float acc = 0.0f;
    for (uint32_t t = 0; t < hist; ++t) {
        acc = fmaf(conv_state[static_cast<size_t>(t) * conv_ch + c], w[t], acc);
    }
    float cur = qkv_mixed[c];
    acc = fmaf(cur, w[hist], acc);
    if (hist > 0u) {
        for (uint32_t t = 0; t + 1u < hist; ++t) {
            conv_state[static_cast<size_t>(t) * conv_ch + c] =
                conv_state[static_cast<size_t>(t + 1u) * conv_ch + c];
        }
        conv_state[static_cast<size_t>(hist - 1u) * conv_ch + c] = cur;
    }
    conv_silu[c] = acc / (1.0f + expf(-acc));
}

// Pre-L2-normalize q_all (mul = 1/sqrt(dk)) and k_all (mul = 1.0): 1 warp per head.
__global__ void k_deltanet_l2norm_qk(
    const float* __restrict__ conv_silu,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    uint32_t n_kh,
    uint32_t dk,
    float eps
) {
    uint32_t warp_id = (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (warp_id >= 2u * n_kh) return;
    bool is_k = (warp_id >= n_kh);
    uint32_t hk = is_k ? (warp_id - n_kh) : warp_id;
    uint32_t qk_dim = n_kh * dk;
    const float* src = conv_silu + (is_k ? qk_dim : 0u) + hk * dk;
    float* dst = (is_k ? k_out : q_out) + hk * dk;
    float mul = is_k ? 1.0f : rsqrtf(float(dk));

    float sq = 0.0f;
    for (uint32_t i = lane; i < dk; i += 32u) {
        float v = src[i];
        sq = fmaf(v, v, sq);
    }
    sq = warp_sum_all(sq);
    float scale = mul * rsqrtf(sq + eps);
    for (uint32_t i = lane; i < dk; i += 32u) {
        dst[i] = src[i] * scale;
    }
}

// 1 warp per (h, jv) row -> n_vh * dv warps (e.g., 2048 warps = 512 blocks of 128 threads!).
// Single read + single write of ssm_state per step via register cache (supports dk <= 512).
__global__ void k_deltanet_recur(
    const float* __restrict__ q_l2,
    const float* __restrict__ k_l2,
    const float* __restrict__ conv_silu,
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ ssm_a,
    const float* __restrict__ ssm_dt_bias,
    float* __restrict__ ssm_state,
    float* __restrict__ o_buf,
    uint32_t n_kh,
    uint32_t n_vh,
    uint32_t dk,
    uint32_t dv
) {
    uint32_t warp_id = (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (warp_id >= n_vh * dv) return;

    uint32_t h = warp_id / dv;
    uint32_t jv = warp_id - h * dv;
    uint32_t hk = h / (n_vh / n_kh);

    const float* qh = q_l2 + hk * dk;
    const float* kh = k_l2 + hk * dk;
    const float* vh = conv_silu + 2u * n_kh * dk + h * dv;

    float beta = 1.0f / (1.0f + expf(-beta_raw[h]));
    float x_sp = alpha_raw[h] + ssm_dt_bias[h];
    float sp = (x_sp > 20.0f) ? x_sp : ((x_sp < -20.0f) ? expf(x_sp) : log1pf(expf(x_sp)));
    float decay = expf(sp * ssm_a[h]);

    float* row = ssm_state + (static_cast<size_t>(h) * dv + jv) * dk;

    float s_reg[16];
    float k_reg[16];
    float sk = 0.0f;
    uint32_t n_it = 0;
    for (uint32_t ik = lane; ik < dk; ik += 32u, ++n_it) {
        float sd = row[ik] * decay;
        float kv = kh[ik];
        s_reg[n_it] = sd;
        k_reg[n_it] = kv;
        sk = fmaf(sd, kv, sk);
    }
    sk = warp_sum_all(sk);
    float d_jv = (vh[jv] - sk) * beta;

    float o_jv = 0.0f;
    n_it = 0;
    for (uint32_t ik = lane; ik < dk; ik += 32u, ++n_it) {
        float sn = fmaf(d_jv, k_reg[n_it], s_reg[n_it]);
        row[ik] = sn;
        o_jv = fmaf(sn, qh[ik], o_jv);
    }
    o_jv = warp_sum_all(o_jv);
    if (lane == 0u) {
        o_buf[h * dv + jv] = o_jv;
    }
}

// Per-head RMSNorm(o_buf) * silu(z) + fused col_sign + FWHT-32 -> xrot_out.
// 1 block per head (128 threads = 4 warps).
__global__ void k_deltanet_norm_gate_fwht32(
    const float* __restrict__ o_buf,
    const float* __restrict__ z,
    const float* __restrict__ ssm_norm,
    float* __restrict__ xrot_out,
    uint32_t n_vh,
    uint32_t dv,
    float eps,
    uint32_t padded_inner
) {
    __shared__ float s_w[4];
    uint32_t h = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid >> 5u;
    uint32_t lane = tid & 31u;

    const float* oh = o_buf + h * dv;
    const float* zh = z + h * dv;
    float sq = 0.0f;
    for (uint32_t j = tid; j < dv; j += 128u) {
        float v = oh[j];
        sq = fmaf(v, v, sq);
    }
    sq = warp_sum_all(sq);
    if (lane == 0u) s_w[wid] = sq;
    __syncthreads();
    if (wid == 0u) {
        float tot = (lane < 4u) ? s_w[lane] : 0.0f;
        tot = warp_sum_all(tot);
        if (lane == 0u) s_w[0] = rsqrtf(tot / float(dv) + eps);
    }
    __syncthreads();
    float inv_rms = s_w[0];

    for (uint32_t base = wid * 32u; base < dv; base += 128u) {
        uint32_t j = base + lane;
        uint32_t idx = h * dv + j;
        float ov = oh[j] * inv_rms * ssm_norm[j];
        float zv = zh[j];
        float yv = ov * (zv / (1.0f + expf(-zv)));
        xrot_out[idx] = warp_fwht32(apply_col_sign(yv, idx), lane);
    }
    if (h + 1u == n_vh) {
        uint32_t inner = n_vh * dv;
        for (uint32_t c = inner + tid; c < padded_inner; c += 128u) {
            xrot_out[c] = 0.0f;
        }
    }
}

// ---------------- Full GQA / Gated-GQA Kernels ----------------

// Split q/q_gate, optional bias, optional q_norm/k_norm, RoPE, append K/V to VRAM cache.
// Grid: nh + nkv blocks, blockDim = 128.
__global__ void k_gqa_prep_kv(
    const float* __restrict__ q_raw,
    const float* __restrict__ k_cur,
    const float* __restrict__ v_cur,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ q_bias,
    const float* __restrict__ k_bias,
    const float* __restrict__ v_bias,
    float* __restrict__ q_out,
    float* __restrict__ q_gate_out,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    uint32_t nh,
    uint32_t nkv,
    uint32_t hd,
    uint32_t pos,
    uint32_t rope_dim,
    float rope_freq_base,
    int rope_neox,
    int gated_q,
    float eps
) {
    __shared__ float s_vec[512];
    __shared__ float s_w[4];
    uint32_t blk = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid >> 5u;
    uint32_t lane = tid & 31u;

    bool is_q = (blk < nh);
    uint32_t h = is_q ? blk : (blk - nh);

    if (is_q) {
        for (uint32_t d = tid; d < hd; d += 128u) {
            if (gated_q) {
                uint32_t iq = h * 2u * hd + d;
                uint32_t ig = iq + hd;
                s_vec[d] = q_raw[iq] + (q_bias ? q_bias[iq] : 0.0f);
                q_gate_out[h * hd + d] = q_raw[ig] + (q_bias ? q_bias[ig] : 0.0f);
            } else {
                uint32_t iq = h * hd + d;
                s_vec[d] = q_raw[iq] + (q_bias ? q_bias[iq] : 0.0f);
            }
        }
    } else {
        size_t kv_off = static_cast<size_t>(pos) * nkv * hd + h * hd;
        for (uint32_t d = tid; d < hd; d += 128u) {
            uint32_t ik = h * hd + d;
            s_vec[d] = k_cur[ik] + (k_bias ? k_bias[ik] : 0.0f);
            v_cache[kv_off + d] = v_cur[ik] + (v_bias ? v_bias[ik] : 0.0f);
        }
    }
    __syncthreads();

    const float* nw = is_q ? q_norm : k_norm;
    if (nw) {
        float sq = 0.0f;
        for (uint32_t d = tid; d < hd; d += 128u) {
            float v = s_vec[d];
            sq = fmaf(v, v, sq);
        }
        sq = warp_sum_all(sq);
        if (lane == 0u) s_w[wid] = sq;
        __syncthreads();
        if (wid == 0u) {
            float tot = (lane < 4u) ? s_w[lane] : 0.0f;
            tot = warp_sum_all(tot);
            if (lane == 0u) s_w[0] = rsqrtf(tot / float(hd) + eps);
        }
        __syncthreads();
        float inv = s_w[0];
        for (uint32_t d = tid; d < hd; d += 128u) {
            s_vec[d] = s_vec[d] * inv * nw[d];
        }
        __syncthreads();
    }

    uint32_t rdim = (rope_dim < hd) ? rope_dim : hd;
    uint32_t half = rdim >> 1u;
    for (uint32_t i = tid; i < half; i += 128u) {
        float freq = powf(rope_freq_base, -(2.0f * float(i)) / float(rdim));
        float theta = float(pos) * freq;
        float sin_t, cos_t;
        sincosf(theta, &sin_t, &cos_t);
        if (rope_neox) {
            float x0 = s_vec[i];
            float x1 = s_vec[i + half];
            s_vec[i] = x0 * cos_t - x1 * sin_t;
            s_vec[i + half] = x0 * sin_t + x1 * cos_t;
        } else {
            float x0 = s_vec[2u * i];
            float x1 = s_vec[2u * i + 1u];
            s_vec[2u * i] = x0 * cos_t - x1 * sin_t;
            s_vec[2u * i + 1u] = x0 * sin_t + x1 * cos_t;
        }
    }
    __syncthreads();

    if (is_q) {
        for (uint32_t d = tid; d < hd; d += 128u) {
            q_out[h * hd + d] = s_vec[d];
        }
    } else {
        size_t kv_off = static_cast<size_t>(pos) * nkv * hd + h * hd;
        for (uint32_t d = tid; d < hd; d += 128u) {
            k_cache[kv_off + d] = s_vec[d];
        }
    }
}

// Online-softmax FlashDecoding GQA + sigmoid(q_gate) + fused col_sign & FWHT-32 -> xrot_out.
// Grid: nh blocks, blockDim = 256 (8 warps). Supports arbitrary seq_len and hd <= 512 (multiple of 32).
__global__ void k_gqa_attn_fwht32(
    const float* __restrict__ q_all,
    const float* __restrict__ q_gate_all,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ xrot_out,
    uint32_t nh,
    uint32_t nkv,
    uint32_t hd,
    uint32_t seq_len,
    int gated_q,
    uint32_t padded_att
) {
    __shared__ float s_q[512];
    __shared__ float s_m[8];
    __shared__ float s_l[8];
    __shared__ float s_acc[8 * 512];

    uint32_t h = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t kv_h = h / (nh / nkv);
    size_t kv_stride = static_cast<size_t>(nkv) * hd;
    float scale = rsqrtf(float(hd));

    for (uint32_t d = tid; d < hd; d += 256u) {
        s_q[d] = q_all[h * hd + d];
    }
    __syncthreads();

    float m_w = -3.402823466e38f;
    float l_w = 0.0f;
    float acc[16];
    uint32_t chunks = hd >> 5u;
    for (uint32_t j = 0; j < chunks; ++j) acc[j] = 0.0f;

    for (uint32_t t = wid; t < seq_len; t += 8u) {
        const float* kh = k_cache + static_cast<size_t>(t) * kv_stride + kv_h * hd;
        const float* vh = v_cache + static_cast<size_t>(t) * kv_stride + kv_h * hd;
        float dot = 0.0f;
        for (uint32_t j = 0; j < chunks; ++j) {
            uint32_t d = (j << 5u) + lane;
            dot = fmaf(s_q[d], kh[d], dot);
        }
        float score = warp_sum_all(dot) * scale;
        if (l_w == 0.0f) {
            m_w = score;
            l_w = 1.0f;
            for (uint32_t j = 0; j < chunks; ++j) {
                acc[j] = vh[(j << 5u) + lane];
            }
        } else {
            float m_new = fmaxf(m_w, score);
            float alpha = expf(m_w - m_new);
            float p = expf(score - m_new);
            l_w = l_w * alpha + p;
            m_w = m_new;
            for (uint32_t j = 0; j < chunks; ++j) {
                uint32_t d = (j << 5u) + lane;
                acc[j] = fmaf(p, vh[d], acc[j] * alpha);
            }
        }
    }

    if (lane == 0u) {
        s_m[wid] = m_w;
        s_l[wid] = l_w;
    }
    for (uint32_t j = 0; j < chunks; ++j) {
        s_acc[wid * 512u + (j << 5u) + lane] = acc[j];
    }
    __syncthreads();

    if (wid == 0u) {
        bool active = (lane < 8u && s_l[lane] > 0.0f);
        float mw = active ? s_m[lane] : -3.402823466e38f;
        float m_all = mw;
        for (int off = 4; off > 0; off >>= 1) {
            m_all = fmaxf(m_all, __shfl_xor_sync(0xffffffffu, m_all, off));
        }
        float ws = active ? expf(mw - m_all) : 0.0f;
        float lw = active ? (s_l[lane] * ws) : 0.0f;
        float l_all = lw;
        for (int off = 4; off > 0; off >>= 1) {
            l_all += __shfl_xor_sync(0xffffffffu, l_all, off);
        }
        if (lane < 8u) {
            s_m[lane] = (l_all > 0.0f) ? (ws / l_all) : 0.0f;
        }
    }
    __syncthreads();

    for (uint32_t j = wid; j < chunks; j += 8u) {
        uint32_t d = (j << 5u) + lane;
        float sum = 0.0f;
        for (uint32_t w = 0; w < 8u; ++w) {
            sum = fmaf(s_acc[w * 512u + d], s_m[w], sum);
        }
        if (gated_q) {
            float gv = q_gate_all[h * hd + d];
            sum *= 1.0f / (1.0f + expf(-gv));
        }
        uint32_t idx = h * hd + d;
        xrot_out[idx] = warp_fwht32(apply_col_sign(sum, idx), lane);
    }
    if (h + 1u == nh) {
        uint32_t att_dim = nh * hd;
        for (uint32_t c = att_dim + tid; c < padded_att; c += 256u) {
            xrot_out[c] = 0.0f;
        }
    }
}

// ---------------- GPU Argmax (with optional presence penalty) ----------------

__global__ void k_argmax_stage1(
    const float* __restrict__ logits,
    const uint8_t* __restrict__ seen,
    float presence_pen,
    uint32_t n_vocab,
    float* __restrict__ block_vals,
    uint32_t* __restrict__ block_idxs
) {
    __shared__ float s_val[8];
    __shared__ uint32_t s_idx[8];
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t stride = gridDim.x * blockDim.x;

    float bv = -3.402823466e38f;
    uint32_t bi = 0u;
    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < n_vocab; i += stride) {
        float v = logits[i];
        if (presence_pen > 0.0f && seen && seen[i]) v -= presence_pen;
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0u) { s_val[wid] = bv; s_idx[wid] = bi; }
    __syncthreads();
    if (wid == 0u) {
        float v = (lane < 8u) ? s_val[lane] : -3.402823466e38f;
        uint32_t idx = (lane < 8u) ? s_idx[lane] : 0xffffffffu;
        for (int off = 4; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, v, off);
            uint32_t oi = __shfl_down_sync(0xffffffffu, idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0u) {
            block_vals[blockIdx.x] = v;
            block_idxs[blockIdx.x] = idx;
        }
    }
}

__global__ void k_argmax_stage2(
    const float* __restrict__ block_vals,
    const uint32_t* __restrict__ block_idxs,
    uint32_t n_blocks,
    uint8_t* __restrict__ seen,
    int mark_seen,
    uint32_t* __restrict__ out_tok
) {
    __shared__ float s_val[8];
    __shared__ uint32_t s_idx[8];
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid >> 5u;
    uint32_t lane = tid & 31u;

    float bv = (tid < n_blocks) ? block_vals[tid] : -3.402823466e38f;
    uint32_t bi = (tid < n_blocks) ? block_idxs[tid] : 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, bv, off);
        uint32_t oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0u) { s_val[wid] = bv; s_idx[wid] = bi; }
    __syncthreads();
    if (wid == 0u) {
        float v = (lane < 8u) ? s_val[lane] : -3.402823466e38f;
        uint32_t idx = (lane < 8u) ? s_idx[lane] : 0xffffffffu;
        for (int off = 4; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, v, off);
            uint32_t oi = __shfl_down_sync(0xffffffffu, idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0u) {
            *out_tok = idx;
            if (mark_seen && seen) seen[idx] = 1u;
        }
    }
}

// ---------------- Host Helpers & Dispatchers ----------------

bool upload_f32_vec(const float* src, size_t n, float** dst) {
    if (!src || n == 0) { *dst = nullptr; return true; }
    if (cudaMalloc(dst, n * sizeof(float)) != cudaSuccess) { *dst = nullptr; return false; }
    if (cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(*dst); *dst = nullptr; return false;
    }
    return true;
}

bool upload_mat(
    DevMat& m,
    int kind,
    uint32_t rows,
    uint32_t cols,
    uint32_t padded_cols,
    const uint16_t* scales,
    size_t n_scales,
    const uint32_t* words,
    size_t n_words,
    const float* f32_data
) {
    m.free_dev();
    m.kind = kind;
    m.rows = rows;
    m.cols = cols;
    m.padded_cols = padded_cols;
    if (kind == MAT_E8 || kind == MAT_E8_HQ) {
        if (cudaMalloc(&m.d_scales, n_scales * sizeof(uint16_t)) != cudaSuccess ||
            cudaMalloc(&m.d_words, n_words * sizeof(uint32_t)) != cudaSuccess ||
            cudaMemcpy(m.d_scales, scales, n_scales * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(m.d_words, words, n_words * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            m.free_dev();
            return false;
        }
        return true;
    } else if (kind == MAT_F32) {
        size_t n = static_cast<size_t>(rows) * cols;
        if (!upload_f32_vec(f32_data, n, &m.d_f32)) { m.free_dev(); return false; }
        return true;
    }
    return kind == MAT_NONE;
}

void launch_gemv(
    const DevMat& m,
    const float* d_x,
    const float* d_xrot,
    float* d_y,
    bool add_res,
    cudaStream_t stream
) {
    if (m.kind == MAT_F32) {
        uint32_t blocks = (m.rows + 3u) >> 2u;
        if (add_res) k_gemv_f32_single<true><<<blocks, 128, 0, stream>>>(m.d_f32, d_x, d_y, m.rows, m.cols);
        else k_gemv_f32_single<false><<<blocks, 128, 0, stream>>>(m.d_f32, d_x, d_y, m.rows, m.cols);
        return;
    }
    uint32_t tiles = (m.rows + 3u) >> 2u;
    uint32_t steps = m.padded_cols >> 8u;
    if (!add_res && m.rows >= 16384u) {
        uint32_t warps = (tiles + 1u) >> 1u;
        uint32_t blocks = (warps + 3u) >> 2u;
        if (m.kind == MAT_E8) {
            k_gemv_turbo_nr8<MAT_E8><<<blocks, 128, 0, stream>>>(
                m.d_scales, m.d_words, d_xrot, d_y, m.rows, m.padded_cols);
        } else {
            k_gemv_turbo_nr8<MAT_E8_HQ><<<blocks, 128, 0, stream>>>(
                m.d_scales, m.d_words, d_xrot, d_y, m.rows, m.padded_cols);
        }
        return;
    }
    // Use split_k on small-row residual-add GEMVs to saturate all SMs with zero memset.
    uint32_t split_k = 1u;
    if (add_res && tiles <= 512u && steps >= 4u) {
        split_k = (steps >= 8u) ? 4u : 2u;
    }
    uint32_t warps_per_block = (tiles * split_k <= 256u) ? 1u : 4u;
    uint32_t blocks_x = (tiles + warps_per_block - 1u) / warps_per_block;
    dim3 grid(blocks_x, split_k);
    uint32_t threads = warps_per_block * 32u;
    if (m.kind == MAT_E8) {
        if (add_res) {
            k_gemv_turbo_sk<MAT_E8, true><<<grid, threads, 0, stream>>>(
                m.d_scales, m.d_words, d_xrot, d_y, m.rows, m.padded_cols, split_k);
        } else {
            k_gemv_turbo_sk<MAT_E8, false><<<grid, threads, 0, stream>>>(
                m.d_scales, m.d_words, d_xrot, d_y, m.rows, m.padded_cols, 1u);
        }
    } else {
        if (add_res) {
            k_gemv_turbo_sk<MAT_E8_HQ, true><<<grid, threads, 0, stream>>>(
                m.d_scales, m.d_words, d_xrot, d_y, m.rows, m.padded_cols, split_k);
        } else {
            k_gemv_turbo_sk<MAT_E8_HQ, false><<<grid, threads, 0, stream>>>(
                m.d_scales, m.d_words, d_xrot, d_y, m.rows, m.padded_cols, 1u);
        }
    }
}

void launch_gemv_pair(
    const DevMat& a, float* ya,
    const DevMat& b, float* yb,
    const float* d_x, const float* d_xrot,
    cudaStream_t stream
) {
    if ((a.kind == MAT_E8 || a.kind == MAT_E8_HQ) && a.kind == b.kind && a.padded_cols == b.padded_cols) {
        uint32_t ta = (a.rows + 3u) >> 2u;
        uint32_t tb = (b.rows + 3u) >> 2u;
        MultiTurboDesc m0{a.d_scales, a.d_words, ya, a.rows, 0u, ta};
        MultiTurboDesc m1{b.d_scales, b.d_words, yb, b.rows, ta, tb};
        MultiTurboDesc m2{};
        uint32_t blocks = (ta + tb + 3u) >> 2u;
        if (a.kind == MAT_E8) {
            k_gemv_turbo_multi<MAT_E8, 2><<<blocks, 128, 0, stream>>>(m0, m1, m2, d_xrot, a.padded_cols);
        } else {
            k_gemv_turbo_multi<MAT_E8_HQ, 2><<<blocks, 128, 0, stream>>>(m0, m1, m2, d_xrot, a.padded_cols);
        }
    } else {
        launch_gemv(a, d_x, d_xrot, ya, false, stream);
        launch_gemv(b, d_x, d_xrot, yb, false, stream);
    }
}

void launch_gemv_triple(
    const DevMat& a, float* ya,
    const DevMat& b, float* yb,
    const DevMat& c, float* yc,
    const float* d_x, const float* d_xrot,
    cudaStream_t stream
) {
    if ((a.kind == MAT_E8 || a.kind == MAT_E8_HQ) && a.kind == b.kind && a.kind == c.kind &&
        a.padded_cols == b.padded_cols && a.padded_cols == c.padded_cols) {
        uint32_t ta = (a.rows + 3u) >> 2u;
        uint32_t tb = (b.rows + 3u) >> 2u;
        uint32_t tc = (c.rows + 3u) >> 2u;
        MultiTurboDesc m0{a.d_scales, a.d_words, ya, a.rows, 0u, ta};
        MultiTurboDesc m1{b.d_scales, b.d_words, yb, b.rows, ta, tb};
        MultiTurboDesc m2{c.d_scales, c.d_words, yc, c.rows, ta + tb, tc};
        uint32_t blocks = (ta + tb + tc + 3u) >> 2u;
        if (a.kind == MAT_E8) {
            k_gemv_turbo_multi<MAT_E8, 3><<<blocks, 128, 0, stream>>>(m0, m1, m2, d_xrot, a.padded_cols);
        } else {
            k_gemv_turbo_multi<MAT_E8_HQ, 3><<<blocks, 128, 0, stream>>>(m0, m1, m2, d_xrot, a.padded_cols);
        }
    } else {
        launch_gemv(a, d_x, d_xrot, ya, false, stream);
        launch_gemv(b, d_x, d_xrot, yb, false, stream);
        launch_gemv(c, d_x, d_xrot, yc, false, stream);
    }
}

bool ensure_kv_cap(CudaLlmState* st, uint32_t needed_pos) {
    if (needed_pos < st->kv_cap) return true;
    const auto* r = st->runner;
    uint32_t new_cap = std::max<uint32_t>(2048u, st->kv_cap * 2u);
    while (new_cap <= needed_pos) new_cap *= 2u;
    size_t kv_stride = static_cast<size_t>(r->n_head_kv) * r->head_dim;
    size_t old_bytes = static_cast<size_t>(st->kv_cap) * kv_stride * sizeof(float);
    size_t new_bytes = static_cast<size_t>(new_cap) * kv_stride * sizeof(float);

    for (uint32_t il = 0; il < r->n_layer; ++il) {
        if (r->layers[il].is_deltanet) continue;
        float* nk = nullptr;
        float* nv = nullptr;
        if (cudaMalloc(&nk, new_bytes) != cudaSuccess || cudaMalloc(&nv, new_bytes) != cudaSuccess) {
            if (nk) cudaFree(nk);
            if (nv) cudaFree(nv);
            return false;
        }
        if (old_bytes > 0) {
            cudaMemcpyAsync(nk, st->d_k_caches[il], old_bytes, cudaMemcpyDeviceToDevice, st->stream);
            cudaMemcpyAsync(nv, st->d_v_caches[il], old_bytes, cudaMemcpyDeviceToDevice, st->stream);
            cudaStreamSynchronize(st->stream);
            cudaFree(st->d_k_caches[il]);
            cudaFree(st->d_v_caches[il]);
        }
        st->d_k_caches[il] = nk;
        st->d_v_caches[il] = nv;
    }
    st->kv_cap = new_cap;
    return true;
}

// Enqueue one complete forward token onto st->stream (zero host syncs!).
bool enqueue_forward_token(
    const CudaLlmRunner* r,
    CudaLlmState* st,
    uint32_t token_id,
    uint32_t pos,
    bool compute_logits
) {
    if (!ensure_kv_cap(st, pos)) return false;
    cudaStream_t s = st->stream;

    // 1. Embedding lookup -> d_cur
    if (r->token_embd.kind == MAT_E8) {
        uint32_t blocks = (r->token_embd.padded_cols + 127u) / 128u;
        k_embed_turbo<MAT_E8><<<blocks, 128, 0, s>>>(
            r->token_embd.d_scales, r->token_embd.d_words,
            token_id, r->n_embd, r->token_embd.padded_cols, r->embed_scale, st->d_cur);
    } else if (r->token_embd.kind == MAT_E8_HQ) {
        uint32_t blocks = (r->token_embd.padded_cols + 127u) / 128u;
        k_embed_turbo<MAT_E8_HQ><<<blocks, 128, 0, s>>>(
            r->token_embd.d_scales, r->token_embd.d_words,
            token_id, r->n_embd, r->token_embd.padded_cols, r->embed_scale, st->d_cur);
    } else {
        uint32_t blocks = (r->n_embd + 127u) / 128u;
        k_embed_f32<<<blocks, 128, 0, s>>>(
            r->token_embd.d_f32, token_id, r->n_embd, r->embed_scale, st->d_cur);
    }

    // 2. Transformer / Hybrid Layers
    for (uint32_t il = 0; il < r->n_layer; ++il) {
        const auto& ly = r->layers[il];

        // Pre-attention RMSNorm + fused FWHT-32
        bool need_normed = ly.is_deltanet || ly.wq.kind == MAT_F32 || ly.wk.kind == MAT_F32 || ly.wv.kind == MAT_F32;
        k_rmsnorm_fwht32<<<1, 256, 0, s>>>(
            st->d_cur, ly.d_attn_norm, r->rms_norm_eps,
            need_normed ? st->d_normed : nullptr, st->d_xrot, r->n_embd, r->padded_embd);

        if (ly.is_deltanet) {
            launch_gemv_pair(ly.wqkv, st->d_qkv, ly.wqkv_gate, st->d_z, st->d_normed, st->d_xrot, s);
            if (ly.ssm_alpha.kind == MAT_F32 && ly.ssm_beta.kind == MAT_F32) {
                k_gemv_f32_pair<<<ly.ssm_alpha.rows + ly.ssm_beta.rows, 128, 0, s>>>(
                    ly.ssm_alpha.d_f32, st->d_alpha, ly.ssm_alpha.rows,
                    ly.ssm_beta.d_f32, st->d_beta, ly.ssm_beta.rows,
                    st->d_normed, r->n_embd);
            } else {
                launch_gemv(ly.ssm_alpha, st->d_normed, st->d_xrot, st->d_alpha, false, s);
                launch_gemv(ly.ssm_beta, st->d_normed, st->d_xrot, st->d_beta, false, s);
            }
            {
                uint32_t blocks = (r->conv_ch + 127u) / 128u;
                k_deltanet_conv_silu<<<blocks, 128, 0, s>>>(
                    st->d_qkv, ly.d_ssm_conv1d, st->d_conv_states[il], st->d_conv_out,
                    r->conv_ch, r->ssm_conv_kernel);
            }
            {
                uint32_t blocks = (2u * r->ssm_group_count + 3u) >> 2u;
                k_deltanet_l2norm_qk<<<blocks, 128, 0, s>>>(
                    st->d_conv_out, st->d_q_l2, st->d_k_l2,
                    r->ssm_group_count, r->ssm_state_size, r->rms_norm_eps);
            }
            {
                uint32_t total_warps = r->ssm_dt_rank * r->dv;
                uint32_t blocks = (total_warps + 3u) >> 2u;
                k_deltanet_recur<<<blocks, 128, 0, s>>>(
                    st->d_q_l2, st->d_k_l2, st->d_conv_out,
                    st->d_alpha, st->d_beta, ly.d_ssm_a, ly.d_ssm_dt_bias,
                    st->d_ssm_states[il], st->d_o_buf,
                    r->ssm_group_count, r->ssm_dt_rank, r->ssm_state_size, r->dv);
            }
            k_deltanet_norm_gate_fwht32<<<r->ssm_dt_rank, 128, 0, s>>>(
                st->d_o_buf, st->d_z, ly.d_ssm_norm, st->d_xrot,
                r->ssm_dt_rank, r->dv, r->rms_norm_eps, ly.ssm_out.padded_cols);

            if (ly.d_attn_post_norm) {
                launch_gemv(ly.ssm_out, nullptr, st->d_xrot, st->d_tmp, false, s);
                k_rmsnorm_add<<<1, 256, 0, s>>>(st->d_tmp, ly.d_attn_post_norm, r->rms_norm_eps, st->d_cur, r->n_embd);
            } else {
                launch_gemv(ly.ssm_out, nullptr, st->d_xrot, st->d_cur, true, s);
            }
        } else {
            launch_gemv_triple(ly.wq, st->d_qraw, ly.wk, st->d_k, ly.wv, st->d_v, st->d_normed, st->d_xrot, s);
            k_gqa_prep_kv<<<r->n_head + r->n_head_kv, 128, 0, s>>>(
                st->d_qraw, st->d_k, st->d_v,
                ly.d_q_norm, ly.d_k_norm, ly.d_q_bias, ly.d_k_bias, ly.d_v_bias,
                st->d_q, st->d_qgate, st->d_k_caches[il], st->d_v_caches[il],
                r->n_head, r->n_head_kv, r->head_dim, pos,
                r->rope_dim, r->rope_freq_base, r->rope_neox ? 1 : 0, ly.gated_q ? 1 : 0, r->rms_norm_eps);
            k_gqa_attn_fwht32<<<r->n_head, 256, 0, s>>>(
                st->d_q, st->d_qgate, st->d_k_caches[il], st->d_v_caches[il],
                st->d_xrot, r->n_head, r->n_head_kv, r->head_dim, pos + 1u,
                ly.gated_q ? 1 : 0, ly.wo.padded_cols);

            if (ly.d_attn_post_norm) {
                launch_gemv(ly.wo, nullptr, st->d_xrot, st->d_tmp, false, s);
                k_rmsnorm_add<<<1, 256, 0, s>>>(st->d_tmp, ly.d_attn_post_norm, r->rms_norm_eps, st->d_cur, r->n_embd);
            } else {
                launch_gemv(ly.wo, nullptr, st->d_xrot, st->d_cur, true, s);
            }
        }

        // Pre-FFN RMSNorm + fused FWHT-32
        k_rmsnorm_fwht32<<<1, 256, 0, s>>>(
            st->d_cur, ly.d_post_attn_norm, r->rms_norm_eps,
            nullptr, st->d_xrot, r->n_embd, r->padded_embd);

        // FFN Gate + Up (fused 2-matrix launch) -> SwiGLU/GeGLU + FWHT-32 -> Down + residual
        launch_gemv_pair(ly.ffn_gate, st->d_gate, ly.ffn_up, st->d_up, nullptr, st->d_xrot, s);
        {
            uint32_t blocks = (r->padded_ff + 127u) / 128u;
            k_swiglu_fwht32<<<blocks, 128, 0, s>>>(
                st->d_gate, st->d_up, st->d_xrot, r->n_ff, r->padded_ff, r->ffn_gelu ? 1 : 0);
        }
        if (ly.d_ffn_post_norm) {
            launch_gemv(ly.ffn_down, nullptr, st->d_xrot, st->d_tmp, false, s);
            k_rmsnorm_add<<<1, 256, 0, s>>>(st->d_tmp, ly.d_ffn_post_norm, r->rms_norm_eps, st->d_cur, r->n_embd);
        } else {
            launch_gemv(ly.ffn_down, nullptr, st->d_xrot, st->d_cur, true, s);
        }
    }

    // 3. Final RMSNorm + LM Head
    if (compute_logits) {
        const DevMat& head = (r->output_head.kind != MAT_NONE) ? r->output_head : r->token_embd;
        k_rmsnorm_fwht32<<<1, 256, 0, s>>>(
            st->d_cur, r->d_output_norm, r->rms_norm_eps,
            (head.kind == MAT_F32) ? st->d_normed : nullptr,
            st->d_xrot, r->n_embd, r->padded_embd);
        launch_gemv(head, st->d_normed, st->d_xrot, st->d_logits, false, s);
    }
    return true;
}

}  // namespace

extern "C" {

struct CudaMatSpecC {
    int kind;
    uint32_t rows;
    uint32_t cols;
    uint32_t padded_cols;
    const uint16_t* scales_f16;
    size_t n_scales;
    const uint32_t* packed_words;
    size_t n_words;
    const float* f32_data;
};

struct CudaLlmCfgC {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_ff;
    uint32_t rope_dim;
    float rope_freq_base;
    uint32_t rope_neox;
    float rms_norm_eps;
    float embed_scale;
    uint32_t ffn_gelu;
    uint32_t ssm_conv_kernel;
    uint32_t ssm_state_size;
    uint32_t ssm_group_count;
    uint32_t ssm_dt_rank;
    uint32_t ssm_inner_size;
};

void* vxt_cuda_llm_create(
    const CudaLlmCfgC* cfg,
    const CudaMatSpecC* token_embd,
    const float* output_norm,
    const CudaMatSpecC* output_head
) {
    if (!cfg || !token_embd || !output_norm) return nullptr;
    if (cudaSetDevice(0) != cudaSuccess) return nullptr;
    auto* r = new CudaLlmRunner();
    r->n_vocab = cfg->n_vocab;
    r->n_embd = cfg->n_embd;
    r->n_layer = cfg->n_layer;
    r->n_head = cfg->n_head;
    r->n_head_kv = cfg->n_head_kv;
    r->head_dim = cfg->head_dim;
    r->n_ff = cfg->n_ff;
    r->rope_dim = cfg->rope_dim;
    r->rope_freq_base = cfg->rope_freq_base;
    r->rope_neox = cfg->rope_neox != 0;
    r->rms_norm_eps = cfg->rms_norm_eps;
    r->embed_scale = cfg->embed_scale;
    r->ffn_gelu = cfg->ffn_gelu != 0;
    r->ssm_conv_kernel = cfg->ssm_conv_kernel;
    r->ssm_state_size = cfg->ssm_state_size;
    r->ssm_group_count = cfg->ssm_group_count;
    r->ssm_dt_rank = cfg->ssm_dt_rank;
    r->ssm_inner_size = cfg->ssm_inner_size;

    r->padded_embd = ((r->n_embd + 255u) / 256u) * 256u;
    r->padded_ff = ((r->n_ff + 255u) / 256u) * 256u;
    r->conv_ch = r->ssm_inner_size + 2u * r->ssm_group_count * r->ssm_state_size;
    r->dv = (r->ssm_dt_rank > 0u) ? (r->ssm_inner_size / r->ssm_dt_rank) : 0u;
    uint32_t padded_att = ((r->n_head * r->head_dim + 255u) / 256u) * 256u;
    uint32_t padded_inner = ((r->ssm_inner_size + 255u) / 256u) * 256u;
    r->max_padded_cols = std::max({r->padded_embd, r->padded_ff, padded_att, padded_inner});
    r->layers.resize(r->n_layer);

    if (!upload_mat(r->token_embd, token_embd->kind, token_embd->rows, token_embd->cols, token_embd->padded_cols,
                    token_embd->scales_f16, token_embd->n_scales, token_embd->packed_words, token_embd->n_words,
                    token_embd->f32_data) ||
        !upload_f32_vec(output_norm, r->n_embd, &r->d_output_norm)) {
        delete r;
        return nullptr;
    }
    if (output_head && output_head->kind != MAT_NONE) {
        if (!upload_mat(r->output_head, output_head->kind, output_head->rows, output_head->cols, output_head->padded_cols,
                        output_head->scales_f16, output_head->n_scales, output_head->packed_words, output_head->n_words,
                        output_head->f32_data)) {
            delete r;
            return nullptr;
        }
    }
    return r;
}

void vxt_cuda_llm_destroy(void* ptr) {
    delete static_cast<CudaLlmRunner*>(ptr);
}

int vxt_cuda_llm_set_layer_deltanet(
    void* runner_ptr,
    uint32_t il,
    const float* attn_norm,
    const float* post_attn_norm,
    const float* attn_post_norm,
    const float* ffn_post_norm,
    const CudaMatSpecC* wqkv,
    const CudaMatSpecC* wqkv_gate,
    const CudaMatSpecC* ssm_alpha,
    const CudaMatSpecC* ssm_beta,
    const CudaMatSpecC* ssm_out,
    const float* ssm_conv1d,
    size_t n_conv1d,
    const float* ssm_a,
    const float* ssm_dt_bias,
    const float* ssm_norm,
    const CudaMatSpecC* ffn_gate,
    const CudaMatSpecC* ffn_up,
    const CudaMatSpecC* ffn_down
) {
    auto* r = static_cast<CudaLlmRunner*>(runner_ptr);
    if (!r || il >= r->n_layer) return -1;
    auto& ly = r->layers[il];
    ly.is_deltanet = true;
    auto up = [](DevMat& d, const CudaMatSpecC* s) {
        return upload_mat(d, s->kind, s->rows, s->cols, s->padded_cols,
                          s->scales_f16, s->n_scales, s->packed_words, s->n_words, s->f32_data);
    };
    if (!upload_f32_vec(attn_norm, r->n_embd, &ly.d_attn_norm) ||
        !upload_f32_vec(post_attn_norm, r->n_embd, &ly.d_post_attn_norm) ||
        !upload_f32_vec(attn_post_norm, attn_post_norm ? r->n_embd : 0, &ly.d_attn_post_norm) ||
        !upload_f32_vec(ffn_post_norm, ffn_post_norm ? r->n_embd : 0, &ly.d_ffn_post_norm) ||
        !up(ly.wqkv, wqkv) || !up(ly.wqkv_gate, wqkv_gate) ||
        !up(ly.ssm_alpha, ssm_alpha) || !up(ly.ssm_beta, ssm_beta) || !up(ly.ssm_out, ssm_out) ||
        !upload_f32_vec(ssm_conv1d, n_conv1d, &ly.d_ssm_conv1d) ||
        !upload_f32_vec(ssm_a, r->ssm_dt_rank, &ly.d_ssm_a) ||
        !upload_f32_vec(ssm_dt_bias, r->ssm_dt_rank, &ly.d_ssm_dt_bias) ||
        !upload_f32_vec(ssm_norm, r->dv, &ly.d_ssm_norm) ||
        !up(ly.ffn_gate, ffn_gate) || !up(ly.ffn_up, ffn_up) || !up(ly.ffn_down, ffn_down)) {
        return -2;
    }
    return 0;
}

int vxt_cuda_llm_set_layer_gqa(
    void* runner_ptr,
    uint32_t il,
    const float* attn_norm,
    const float* post_attn_norm,
    const float* attn_post_norm,
    const float* ffn_post_norm,
    const CudaMatSpecC* wq,
    const CudaMatSpecC* wk,
    const CudaMatSpecC* wv,
    const CudaMatSpecC* wo,
    const float* q_norm,
    const float* k_norm,
    const float* q_bias,
    size_t n_q_bias,
    const float* k_bias,
    size_t n_k_bias,
    const float* v_bias,
    size_t n_v_bias,
    uint32_t gated_q,
    const CudaMatSpecC* ffn_gate,
    const CudaMatSpecC* ffn_up,
    const CudaMatSpecC* ffn_down
) {
    auto* r = static_cast<CudaLlmRunner*>(runner_ptr);
    if (!r || il >= r->n_layer) return -1;
    auto& ly = r->layers[il];
    ly.is_deltanet = false;
    ly.gated_q = gated_q != 0;
    auto up = [](DevMat& d, const CudaMatSpecC* s) {
        return upload_mat(d, s->kind, s->rows, s->cols, s->padded_cols,
                          s->scales_f16, s->n_scales, s->packed_words, s->n_words, s->f32_data);
    };
    if (!upload_f32_vec(attn_norm, r->n_embd, &ly.d_attn_norm) ||
        !upload_f32_vec(post_attn_norm, r->n_embd, &ly.d_post_attn_norm) ||
        !upload_f32_vec(attn_post_norm, attn_post_norm ? r->n_embd : 0, &ly.d_attn_post_norm) ||
        !upload_f32_vec(ffn_post_norm, ffn_post_norm ? r->n_embd : 0, &ly.d_ffn_post_norm) ||
        !up(ly.wq, wq) || !up(ly.wk, wk) || !up(ly.wv, wv) || !up(ly.wo, wo) ||
        !upload_f32_vec(q_norm, q_norm ? r->head_dim : 0, &ly.d_q_norm) ||
        !upload_f32_vec(k_norm, k_norm ? r->head_dim : 0, &ly.d_k_norm) ||
        !upload_f32_vec(q_bias, n_q_bias, &ly.d_q_bias) ||
        !upload_f32_vec(k_bias, n_k_bias, &ly.d_k_bias) ||
        !upload_f32_vec(v_bias, n_v_bias, &ly.d_v_bias) ||
        !up(ly.ffn_gate, ffn_gate) || !up(ly.ffn_up, ffn_up) || !up(ly.ffn_down, ffn_down)) {
        return -2;
    }
    return 0;
}

void* vxt_cuda_llm_state_create(const void* runner_ptr) {
    const auto* r = static_cast<const CudaLlmRunner*>(runner_ptr);
    if (!r) return nullptr;
    auto* st = new CudaLlmState();
    st->runner = r;
    if (cudaStreamCreate(&st->stream) != cudaSuccess) { delete st; return nullptr; }

    auto alloc_f = [](float** p, size_t n) -> bool {
        if (n == 0) { *p = nullptr; return true; }
        return cudaMalloc(p, n * sizeof(float)) == cudaSuccess;
    };
    size_t att_dim = static_cast<size_t>(r->n_head) * r->head_dim;
    size_t kv_dim = static_cast<size_t>(r->n_head_kv) * r->head_dim;
    size_t qk_dim = static_cast<size_t>(r->ssm_group_count) * r->ssm_state_size;

    if (!alloc_f(&st->d_cur, r->n_embd) ||
        !alloc_f(&st->d_normed, r->n_embd) ||
        !alloc_f(&st->d_xrot, r->max_padded_cols) ||
        !alloc_f(&st->d_tmp, r->n_embd) ||
        !alloc_f(&st->d_qkv, r->conv_ch) ||
        !alloc_f(&st->d_z, r->ssm_inner_size) ||
        !alloc_f(&st->d_alpha, r->ssm_dt_rank) ||
        !alloc_f(&st->d_beta, r->ssm_dt_rank) ||
        !alloc_f(&st->d_conv_out, r->conv_ch) ||
        !alloc_f(&st->d_q_l2, qk_dim) ||
        !alloc_f(&st->d_k_l2, qk_dim) ||
        !alloc_f(&st->d_o_buf, r->ssm_inner_size) ||
        !alloc_f(&st->d_qraw, att_dim * 2u) ||
        !alloc_f(&st->d_q, att_dim) ||
        !alloc_f(&st->d_qgate, att_dim) ||
        !alloc_f(&st->d_k, kv_dim) ||
        !alloc_f(&st->d_v, kv_dim) ||
        !alloc_f(&st->d_gate, r->n_ff) ||
        !alloc_f(&st->d_up, r->n_ff) ||
        !alloc_f(&st->d_logits, r->n_vocab) ||
        cudaMalloc(&st->d_seen, r->n_vocab) != cudaSuccess ||
        !alloc_f(&st->d_arg_val, 256) ||
        cudaMalloc(&st->d_arg_idx, 256 * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&st->d_next_tok, sizeof(uint32_t)) != cudaSuccess ||
        cudaHostAlloc(&st->h_next_tok, sizeof(uint32_t), cudaHostAllocDefault) != cudaSuccess) {
        delete st;
        return nullptr;
    }
    cudaMemsetAsync(st->d_seen, 0, r->n_vocab, st->stream);

    st->d_conv_states.assign(r->n_layer, nullptr);
    st->d_ssm_states.assign(r->n_layer, nullptr);
    st->d_k_caches.assign(r->n_layer, nullptr);
    st->d_v_caches.assign(r->n_layer, nullptr);

    size_t conv_hist = (r->ssm_conv_kernel > 0u) ? static_cast<size_t>(r->ssm_conv_kernel - 1u) * r->conv_ch : 0;
    size_t ssm_len = static_cast<size_t>(r->ssm_dt_rank) * r->dv * r->ssm_state_size;
    for (uint32_t il = 0; il < r->n_layer; ++il) {
        if (r->layers[il].is_deltanet) {
            if (!alloc_f(&st->d_conv_states[il], conv_hist) || !alloc_f(&st->d_ssm_states[il], ssm_len)) {
                delete st;
                return nullptr;
            }
            if (conv_hist > 0) cudaMemsetAsync(st->d_conv_states[il], 0, conv_hist * sizeof(float), st->stream);
            if (ssm_len > 0) cudaMemsetAsync(st->d_ssm_states[il], 0, ssm_len * sizeof(float), st->stream);
        }
    }
    if (!ensure_kv_cap(st, 0)) { delete st; return nullptr; }
    cudaStreamSynchronize(st->stream);
    return st;
}

void vxt_cuda_llm_state_destroy(void* st_ptr) {
    delete static_cast<CudaLlmState*>(st_ptr);
}

int vxt_cuda_llm_step(
    const void* runner_ptr,
    void* st_ptr,
    uint32_t token_id,
    uint32_t pos,
    float* logits_host_or_null
) {
    const auto* r = static_cast<const CudaLlmRunner*>(runner_ptr);
    auto* st = static_cast<CudaLlmState*>(st_ptr);
    if (!r || !st) return -1;
    bool want_logits = (logits_host_or_null != nullptr);
    if (!enqueue_forward_token(r, st, token_id, pos, want_logits)) return -2;
    if (want_logits) {
        if (cudaMemcpyAsync(logits_host_or_null, st->d_logits,
                            static_cast<size_t>(r->n_vocab) * sizeof(float),
                            cudaMemcpyDeviceToHost, st->stream) != cudaSuccess) {
            return -3;
        }
        if (cudaStreamSynchronize(st->stream) != cudaSuccess) return -4;
    }
    return 0;
}

int vxt_cuda_llm_prefill(
    const void* runner_ptr,
    void* st_ptr,
    const uint32_t* tokens,
    size_t n_tokens,
    uint32_t start_pos,
    float* logits_host
) {
    const auto* r = static_cast<const CudaLlmRunner*>(runner_ptr);
    auto* st = static_cast<CudaLlmState*>(st_ptr);
    if (!r || !st || !tokens || n_tokens == 0 || !logits_host) return -1;
    if (!ensure_kv_cap(st, start_pos + static_cast<uint32_t>(n_tokens))) return -2;
    for (size_t i = 0; i < n_tokens; ++i) {
        bool last = (i + 1 == n_tokens);
        if (!enqueue_forward_token(r, st, tokens[i], start_pos + static_cast<uint32_t>(i), last)) {
            return -3;
        }
    }
    if (cudaMemcpyAsync(logits_host, st->d_logits,
                        static_cast<size_t>(r->n_vocab) * sizeof(float),
                        cudaMemcpyDeviceToHost, st->stream) != cudaSuccess) {
        return -4;
    }
    if (cudaStreamSynchronize(st->stream) != cudaSuccess) return -5;
    return 0;
}

int vxt_cuda_llm_init_seen(
    const void* runner_ptr,
    void* st_ptr,
    const uint32_t* tok_ids,
    size_t n_toks
) {
    const auto* r = static_cast<const CudaLlmRunner*>(runner_ptr);
    auto* st = static_cast<CudaLlmState*>(st_ptr);
    if (!r || !st) return -1;
    cudaMemsetAsync(st->d_seen, 0, r->n_vocab, st->stream);
    for (size_t i = 0; i < n_toks; ++i) {
        if (tok_ids[i] < r->n_vocab) {
            cudaMemsetAsync(st->d_seen + tok_ids[i], 1, 1, st->stream);
        }
    }
    return 0;
}

int vxt_cuda_llm_step_greedy(
    const void* runner_ptr,
    void* st_ptr,
    uint32_t token_id,
    uint32_t pos,
    float presence_pen,
    uint32_t* out_next_token
) {
    const auto* r = static_cast<const CudaLlmRunner*>(runner_ptr);
    auto* st = static_cast<CudaLlmState*>(st_ptr);
    if (!r || !st || !out_next_token) return -1;
    if (!enqueue_forward_token(r, st, token_id, pos, true)) return -2;

    uint32_t n_blocks = 256u;
    k_argmax_stage1<<<n_blocks, 256, 0, st->stream>>>(
        st->d_logits, st->d_seen, presence_pen, r->n_vocab, st->d_arg_val, st->d_arg_idx);
    k_argmax_stage2<<<1, 256, 0, st->stream>>>(
        st->d_arg_val, st->d_arg_idx, n_blocks, st->d_seen, (presence_pen > 0.0f) ? 1 : 0, st->d_next_tok);
    if (cudaMemcpyAsync(st->h_next_tok, st->d_next_tok, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, st->stream) != cudaSuccess) {
        return -3;
    }
    if (cudaStreamSynchronize(st->stream) != cudaSuccess) return -4;
    *out_next_token = *st->h_next_tok;
    return 0;
}

size_t vxt_cuda_llm_kv_bytes(const void* st_ptr) {
    const auto* st = static_cast<const CudaLlmState*>(st_ptr);
    if (!st || !st->runner) return 0;
    const auto* r = st->runner;
    size_t gqa_layers = 0;
    for (const auto& ly : r->layers) {
        if (!ly.is_deltanet) gqa_layers++;
    }
    return gqa_layers * 2u * static_cast<size_t>(st->kv_cap) * r->n_head_kv * r->head_dim * sizeof(float);
}

}  // extern "C"
