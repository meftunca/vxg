// Native NVIDIA CUDA zero-LUT E8 (bits=7), E8-HQ (bits=8), and RawF32 GEMV kernels
// for VXT-Turbo / VXT-Turbo-HQ archives. Compiled when VXG_CUDA=1 on Linux/NVIDIA.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <mutex>

namespace {

constexpr int KIND_TURBO_E8 = 7;
constexpr int KIND_TURBO_E8_HQ = 8;
constexpr int KIND_RAW_F32 = 1;

__device__ __forceinline__ uint32_t vxt_col_sign_mask(uint32_t col) {
    uint32_t h = col ^ 0x9e3779b9u;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return (h & 1u) << 31u;
}

__device__ __forceinline__ float vxt_apply_col_sign(float v, uint32_t col) {
    if (v == 0.0f) return 0.0f;
    uint32_t bits = __float_as_uint(v) ^ vxt_col_sign_mask(col);
    return __uint_as_float(bits);
}

// Warp-shuffle FWHT-32 + column sign flip (1 warp = 32 columns, zero shared memory).
__global__ void vxt_cuda_fwht32_rot(
    const float* __restrict__ x_in,
    float* __restrict__ x_rot_out,
    uint32_t cols,
    uint32_t padded_cols
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= padded_cols) return;
    uint32_t lane = threadIdx.x & 31u;
    float v = (idx < cols) ? vxt_apply_col_sign(x_in[idx], idx) : 0.0f;
    for (uint32_t mask = 1u; mask < 32u; mask <<= 1u) {
        float other = __shfl_xor_sync(0xffffffffu, v, mask);
        v = ((lane & mask) == 0u) ? (v + other) : (other - v);
    }
    x_rot_out[idx] = v * 0.1767766952966369f;  // 1 / sqrt(32)
}

__device__ __forceinline__ float vxt_dot_e8_row(uint32_t rw, float4 x0, float4 x1) {
    uint32_t m7 = rw & 0x0fffffffu;
    uint32_t par = m7 ^ (m7 >> 4u);
    par ^= par >> 8u;
    uint32_t u7 = ((rw >> 27u) & 14u) | ((par ^ (par >> 16u)) & 1u);
    float bias = (rw & 0x80000000u) ? -7.5f : -8.0f;

    float q0 = float(rw & 0xfu) + bias;
    float q1 = float((rw >> 4u) & 0xfu) + bias;
    float q2 = float((rw >> 8u) & 0xfu) + bias;
    float q3 = float((rw >> 12u) & 0xfu) + bias;
    float q4 = float((rw >> 16u) & 0xfu) + bias;
    float q5 = float((rw >> 20u) & 0xfu) + bias;
    float q6 = float((rw >> 24u) & 0xfu) + bias;
    float q7 = float(u7) + bias;

    float s0 = fmaf(q0, x0.x, fmaf(q1, x0.y, fmaf(q2, x0.z, q3 * x0.w)));
    float s1 = fmaf(q4, x1.x, fmaf(q5, x1.y, fmaf(q6, x1.z, q7 * x1.w)));
    return s0 + s1;
}

__device__ __forceinline__ float vxt_dot_e8_hq_row(uint32_t w0, uint32_t w1, float4 x0, float4 x1) {
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

__device__ __forceinline__ float4 vxt_load_half4_f32(const uint16_t* p) {
    const __half* hp = reinterpret_cast<const __half*>(p);
    return make_float4(__half2float(hp[0]), __half2float(hp[1]), __half2float(hp[2]), __half2float(hp[3]));
}

__device__ __forceinline__ float4 vxt_warp_sum4(float4 v) {
    for (int off = 16; off > 0; off >>= 1) {
        v.x += __shfl_down_sync(0xffffffffu, v.x, off);
        v.y += __shfl_down_sync(0xffffffffu, v.y, off);
        v.z += __shfl_down_sync(0xffffffffu, v.z, off);
        v.w += __shfl_down_sync(0xffffffffu, v.w, off);
    }
    return v;
}

__device__ __forceinline__ void vxt_store_tile4(float* y, uint32_t rows, uint32_t tile, float4 s) {
    uint32_t r = tile << 2u;
    if (r + 3u < rows) {
        reinterpret_cast<float4*>(y + r)[0] = s;
    } else {
        if (r < rows) y[r] = s.x;
        if (r + 1u < rows) y[r + 1u] = s.y;
        if (r + 2u < rows) y[r + 2u] = s.z;
    }
}

// 4 warps/block, 2 tiles (8 output rows) per warp for high x_rot reuse.
__global__ void vxt_cuda_gemv_e8(
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
    uint32_t b8 = lane >> 2u;

    const uint16_t* sc0 = scales_f16 + static_cast<size_t>(tile0) * steps * 32u;
    const uint32_t* pw0 = packed_words + static_cast<size_t>(tile0) * steps * 128u;
    float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (has_t1) {
        const uint16_t* sc1 = sc0 + static_cast<size_t>(steps) * 32u;
        const uint32_t* pw1 = pw0 + static_cast<size_t>(steps) * 128u;
        for (uint32_t step = 0; step < steps; ++step) {
            const float4* xp = reinterpret_cast<const float4*>(x_rot + (step << 8u) + (lane << 3u));
            float4 x0 = xp[0], x1 = xp[1];
            uint32_t soff = (step << 5u) + (b8 << 2u);
            uint4 w0 = reinterpret_cast<const uint4*>(pw0 + step * 128u)[lane];
            uint4 w1 = reinterpret_cast<const uint4*>(pw1 + step * 128u)[lane];
            float4 sh0 = vxt_load_half4_f32(sc0 + soff);
            float4 sh1 = vxt_load_half4_f32(sc1 + soff);
            acc0.x = fmaf(sh0.x, vxt_dot_e8_row(w0.x, x0, x1), acc0.x);
            acc0.y = fmaf(sh0.y, vxt_dot_e8_row(w0.y, x0, x1), acc0.y);
            acc0.z = fmaf(sh0.z, vxt_dot_e8_row(w0.z, x0, x1), acc0.z);
            acc0.w = fmaf(sh0.w, vxt_dot_e8_row(w0.w, x0, x1), acc0.w);
            acc1.x = fmaf(sh1.x, vxt_dot_e8_row(w1.x, x0, x1), acc1.x);
            acc1.y = fmaf(sh1.y, vxt_dot_e8_row(w1.y, x0, x1), acc1.y);
            acc1.z = fmaf(sh1.z, vxt_dot_e8_row(w1.z, x0, x1), acc1.z);
            acc1.w = fmaf(sh1.w, vxt_dot_e8_row(w1.w, x0, x1), acc1.w);
        }
    } else {
        for (uint32_t step = 0; step < steps; ++step) {
            const float4* xp = reinterpret_cast<const float4*>(x_rot + (step << 8u) + (lane << 3u));
            float4 x0 = xp[0], x1 = xp[1];
            uint4 w0 = reinterpret_cast<const uint4*>(pw0 + step * 128u)[lane];
            float4 sh0 = vxt_load_half4_f32(sc0 + (step << 5u) + (b8 << 2u));
            acc0.x = fmaf(sh0.x, vxt_dot_e8_row(w0.x, x0, x1), acc0.x);
            acc0.y = fmaf(sh0.y, vxt_dot_e8_row(w0.y, x0, x1), acc0.y);
            acc0.z = fmaf(sh0.z, vxt_dot_e8_row(w0.z, x0, x1), acc0.z);
            acc0.w = fmaf(sh0.w, vxt_dot_e8_row(w0.w, x0, x1), acc0.w);
        }
    }

    float4 s0 = vxt_warp_sum4(acc0);
    float4 s1 = vxt_warp_sum4(acc1);
    if (lane == 0u) {
        vxt_store_tile4(y_out, rows, tile0, s0);
        if (has_t1) vxt_store_tile4(y_out, rows, tile0 + 1u, s1);
    }
}

__global__ void vxt_cuda_gemv_e8_hq(
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
    uint32_t b8 = lane >> 2u;

    const uint16_t* sc0 = scales_f16 + static_cast<size_t>(tile0) * steps * 32u;
    const uint32_t* pw0 = packed_words + static_cast<size_t>(tile0) * steps * 256u;
    float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (has_t1) {
        const uint16_t* sc1 = sc0 + static_cast<size_t>(steps) * 32u;
        const uint32_t* pw1 = pw0 + static_cast<size_t>(steps) * 256u;
        for (uint32_t step = 0; step < steps; ++step) {
            const float4* xp = reinterpret_cast<const float4*>(x_rot + (step << 8u) + (lane << 3u));
            float4 x0 = xp[0], x1 = xp[1];
            uint32_t soff = (step << 5u) + (b8 << 2u);
            const uint4* p0 = reinterpret_cast<const uint4*>(pw0 + step * 256u);
            const uint4* p1 = reinterpret_cast<const uint4*>(pw1 + step * 256u);
            uint4 wa0 = p0[lane * 2u], wb0 = p0[lane * 2u + 1u];
            uint4 wa1 = p1[lane * 2u], wb1 = p1[lane * 2u + 1u];
            float4 sh0 = vxt_load_half4_f32(sc0 + soff);
            float4 sh1 = vxt_load_half4_f32(sc1 + soff);
            acc0.x = fmaf(sh0.x, vxt_dot_e8_hq_row(wa0.x, wa0.y, x0, x1), acc0.x);
            acc0.y = fmaf(sh0.y, vxt_dot_e8_hq_row(wa0.z, wa0.w, x0, x1), acc0.y);
            acc0.z = fmaf(sh0.z, vxt_dot_e8_hq_row(wb0.x, wb0.y, x0, x1), acc0.z);
            acc0.w = fmaf(sh0.w, vxt_dot_e8_hq_row(wb0.z, wb0.w, x0, x1), acc0.w);
            acc1.x = fmaf(sh1.x, vxt_dot_e8_hq_row(wa1.x, wa1.y, x0, x1), acc1.x);
            acc1.y = fmaf(sh1.y, vxt_dot_e8_hq_row(wa1.z, wa1.w, x0, x1), acc1.y);
            acc1.z = fmaf(sh1.z, vxt_dot_e8_hq_row(wb1.x, wb1.y, x0, x1), acc1.z);
            acc1.w = fmaf(sh1.w, vxt_dot_e8_hq_row(wb1.z, wb1.w, x0, x1), acc1.w);
        }
    } else {
        for (uint32_t step = 0; step < steps; ++step) {
            const float4* xp = reinterpret_cast<const float4*>(x_rot + (step << 8u) + (lane << 3u));
            float4 x0 = xp[0], x1 = xp[1];
            const uint4* p0 = reinterpret_cast<const uint4*>(pw0 + step * 256u);
            uint4 wa0 = p0[lane * 2u], wb0 = p0[lane * 2u + 1u];
            float4 sh0 = vxt_load_half4_f32(sc0 + (step << 5u) + (b8 << 2u));
            acc0.x = fmaf(sh0.x, vxt_dot_e8_hq_row(wa0.x, wa0.y, x0, x1), acc0.x);
            acc0.y = fmaf(sh0.y, vxt_dot_e8_hq_row(wa0.z, wa0.w, x0, x1), acc0.y);
            acc0.z = fmaf(sh0.z, vxt_dot_e8_hq_row(wb0.x, wb0.y, x0, x1), acc0.z);
            acc0.w = fmaf(sh0.w, vxt_dot_e8_hq_row(wb0.z, wb0.w, x0, x1), acc0.w);
        }
    }

    float4 s0 = vxt_warp_sum4(acc0);
    float4 s1 = vxt_warp_sum4(acc1);
    if (lane == 0u) {
        vxt_store_tile4(y_out, rows, tile0, s0);
        if (has_t1) vxt_store_tile4(y_out, rows, tile0 + 1u, s1);
    }
}

__global__ void vxt_cuda_gemv_f32(
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
    if (lane == 0u) y[row] = sum;
}

struct CudaCtx {
    cudaStream_t stream = nullptr;
    float* d_x = nullptr;
    size_t cap_x = 0;
    float* d_xrot = nullptr;
    size_t cap_xrot = 0;
    float* d_y = nullptr;
    size_t cap_y = 0;
    std::mutex mu;

    bool ensure(size_t nx, size_t nxrot, size_t ny) {
        if (nx > cap_x) {
            if (d_x) cudaFree(d_x);
            if (cudaMalloc(&d_x, nx * sizeof(float)) != cudaSuccess) { d_x = nullptr; cap_x = 0; return false; }
            cap_x = nx;
        }
        if (nxrot > cap_xrot) {
            if (d_xrot) cudaFree(d_xrot);
            if (cudaMalloc(&d_xrot, nxrot * sizeof(float)) != cudaSuccess) { d_xrot = nullptr; cap_xrot = 0; return false; }
            cap_xrot = nxrot;
        }
        if (ny > cap_y) {
            if (d_y) cudaFree(d_y);
            if (cudaMalloc(&d_y, ny * sizeof(float)) != cudaSuccess) { d_y = nullptr; cap_y = 0; return false; }
            cap_y = ny;
        }
        return true;
    }
};

struct CudaTensor {
    int kind = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t padded_cols = 0;
    uint16_t* d_scales = nullptr;
    uint32_t* d_words = nullptr;
    float* d_f32 = nullptr;
};

void launch_one(CudaCtx* ctx, const CudaTensor* t, float* d_y_slot) {
    if (t->kind == KIND_TURBO_E8 || t->kind == KIND_TURBO_E8_HQ) {
        uint32_t tiles = (t->rows + 3u) >> 2u;
        uint32_t warps = (tiles + 1u) >> 1u;
        uint32_t blocks = (warps + 3u) >> 2u;
        if (t->kind == KIND_TURBO_E8) {
            vxt_cuda_gemv_e8<<<blocks, 128, 0, ctx->stream>>>(
                t->d_scales, t->d_words, ctx->d_xrot, d_y_slot, t->rows, t->padded_cols);
        } else {
            vxt_cuda_gemv_e8_hq<<<blocks, 128, 0, ctx->stream>>>(
                t->d_scales, t->d_words, ctx->d_xrot, d_y_slot, t->rows, t->padded_cols);
        }
    } else if (t->kind == KIND_RAW_F32) {
        uint32_t blocks = (t->rows + 3u) >> 2u;
        vxt_cuda_gemv_f32<<<blocks, 128, 0, ctx->stream>>>(
            t->d_f32, ctx->d_x, d_y_slot, t->rows, t->cols);
    }
}

}  // namespace

extern "C" {

int vxt_cuda_device_count() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

int vxt_cuda_device_name(int dev, char* buf, size_t cap) {
    if (!buf || cap == 0) return -1;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return -1;
    snprintf(buf, cap, "%s", prop.name);
    return 0;
}

void* vxt_cuda_ctx_create() {
    if (vxt_cuda_device_count() <= 0) return nullptr;
    if (cudaSetDevice(0) != cudaSuccess) return nullptr;
    auto* ctx = new CudaCtx();
    if (cudaStreamCreate(&ctx->stream) != cudaSuccess) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void vxt_cuda_ctx_destroy(void* p) {
    if (!p) return;
    auto* ctx = static_cast<CudaCtx*>(p);
    if (ctx->d_x) cudaFree(ctx->d_x);
    if (ctx->d_xrot) cudaFree(ctx->d_xrot);
    if (ctx->d_y) cudaFree(ctx->d_y);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    delete ctx;
}

void* vxt_cuda_upload_turbo(
    void* /*ctx*/,
    uint8_t bits,
    uint32_t rows,
    uint32_t cols,
    uint32_t padded_cols,
    const uint16_t* scales_f16,
    size_t n_scales,
    const uint32_t* packed_words,
    size_t n_words
) {
    if (bits != 7 && bits != 8) return nullptr;
    auto* t = new CudaTensor();
    t->kind = (bits == 7) ? KIND_TURBO_E8 : KIND_TURBO_E8_HQ;
    t->rows = rows;
    t->cols = cols;
    t->padded_cols = padded_cols;
    if (cudaMalloc(&t->d_scales, n_scales * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&t->d_words, n_words * sizeof(uint32_t)) != cudaSuccess ||
        cudaMemcpy(t->d_scales, scales_f16, n_scales * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t->d_words, packed_words, n_words * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        if (t->d_scales) cudaFree(t->d_scales);
        if (t->d_words) cudaFree(t->d_words);
        delete t;
        return nullptr;
    }
    return t;
}

void* vxt_cuda_upload_f32(void* /*ctx*/, uint32_t rows, uint32_t cols, const float* data) {
    auto* t = new CudaTensor();
    t->kind = KIND_RAW_F32;
    t->rows = rows;
    t->cols = cols;
    t->padded_cols = cols;
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    if (cudaMalloc(&t->d_f32, bytes) != cudaSuccess ||
        cudaMemcpy(t->d_f32, data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (t->d_f32) cudaFree(t->d_f32);
        delete t;
        return nullptr;
    }
    return t;
}

void vxt_cuda_free_tensor(void* p) {
    if (!p) return;
    auto* t = static_cast<CudaTensor*>(p);
    if (t->d_scales) cudaFree(t->d_scales);
    if (t->d_words) cudaFree(t->d_words);
    if (t->d_f32) cudaFree(t->d_f32);
    delete t;
}

int vxt_cuda_gemv_multi(
    void* ctx_p,
    size_t n_jobs,
    void* const* tensors,
    const float* x_host,
    float* const* y_hosts
) {
    if (!ctx_p || n_jobs == 0 || !tensors || !x_host || !y_hosts) return -1;
    auto* ctx = static_cast<CudaCtx*>(ctx_p);
    std::lock_guard<std::mutex> lk(ctx->mu);

    uint32_t max_cols = 0, max_pcols = 0;
    size_t total_y = 0;
    bool need_rot = false;
    for (size_t i = 0; i < n_jobs; ++i) {
        auto* t = static_cast<const CudaTensor*>(tensors[i]);
        if (!t) return -2;
        max_cols = std::max(max_cols, t->cols);
        max_pcols = std::max(max_pcols, t->padded_cols);
        total_y += t->rows;
        if (t->kind == KIND_TURBO_E8 || t->kind == KIND_TURBO_E8_HQ) need_rot = true;
    }
    if (!ctx->ensure(max_cols, need_rot ? max_pcols : 1u, total_y)) return -3;

    if (cudaMemcpyAsync(ctx->d_x, x_host, max_cols * sizeof(float), cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess) {
        return -4;
    }
    if (need_rot) {
        uint32_t blocks = (max_pcols + 127u) >> 7u;
        vxt_cuda_fwht32_rot<<<blocks, 128, 0, ctx->stream>>>(ctx->d_x, ctx->d_xrot, max_cols, max_pcols);
    }

    size_t y_off = 0;
    for (size_t i = 0; i < n_jobs; ++i) {
        auto* t = static_cast<const CudaTensor*>(tensors[i]);
        launch_one(ctx, t, ctx->d_y + y_off);
        y_off += t->rows;
    }

    y_off = 0;
    for (size_t i = 0; i < n_jobs; ++i) {
        auto* t = static_cast<const CudaTensor*>(tensors[i]);
        if (cudaMemcpyAsync(y_hosts[i], ctx->d_y + y_off, t->rows * sizeof(float), cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess) {
            return -5;
        }
        y_off += t->rows;
    }
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess) return -6;
    return 0;
}

int vxt_cuda_gemv(void* ctx_p, void* tensor, const float* x_host, float* y_host) {
    void* ts[1] = {tensor};
    float* ys[1] = {y_host};
    return vxt_cuda_gemv_multi(ctx_p, 1, ts, x_host, ys);
}

}  // extern "C"
