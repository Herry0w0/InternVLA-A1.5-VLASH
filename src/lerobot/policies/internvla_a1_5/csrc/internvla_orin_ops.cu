// Fused SM87 (Jetson AGX Orin) BF16 CUDA kernels for InternVLA-A1.5 FlashRT backend.
// Pure CUDA + pybind11 (raw device pointers, zero PyTorch C++ ABI dependency, 100% CUDA-Graph safe).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cmath>

namespace py = pybind11;

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    const int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// ---------------------------------------------------------------------------
// 1. Fused split SiLU * Up directly from fused [M, 2 * inter] buffer
// ---------------------------------------------------------------------------
__global__ void silu_mul_split_bf16_kernel(
    const __nv_bfloat16* __restrict__ gu,
    __nv_bfloat16* __restrict__ out,
    int M,
    int inter)
{
    const int total = M * inter;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int m = idx / inter;
    const int i = idx - m * inter;
    const int base = m * (2 * inter) + i;
    const float g = bf16_to_f32(gu[base]);
    const float u = bf16_to_f32(gu[base + inter]);
    const float s = g / (1.0f + __expf(-g));
    out[idx] = f32_to_bf16(s * u);
}

// ---------------------------------------------------------------------------
// 2. Vectorized split of [M, cdim + vdim] -> qkv [M, cdim], z [M, vdim]
//    (cdim and vdim must be multiples of 8 bf16 = 16 bytes)
// ---------------------------------------------------------------------------
__global__ void split_qkvz_bf16_vec8_kernel(
    const int4* __restrict__ src,
    int4* __restrict__ qkv_out,
    int4* __restrict__ z_out,
    int M,
    int cdim_v8,
    int vdim_v8,
    int stride_v8)
{
    const int row = blockIdx.y;
    if (row >= M) return;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int4* row_ptr = src + (size_t)row * stride_v8;
    if (col < cdim_v8) {
        qkv_out[(size_t)row * cdim_v8 + col] = row_ptr[col];
    } else if (col < cdim_v8 + vdim_v8) {
        z_out[(size_t)row * vdim_v8 + (col - cdim_v8)] = row_ptr[col];
    }
}

// ---------------------------------------------------------------------------
// 3a. GDN Q/K L2Norm + V split directly from conv_out [S, 2*H*D + H*D]
//     Grid: (H, S), Block: 128 (assumes D == 128)
// ---------------------------------------------------------------------------
__global__ void gdn_split_l2norm_d128_bf16_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    __nv_bfloat16* __restrict__ q_l2,
    __nv_bfloat16* __restrict__ k_l2,
    __nv_bfloat16* __restrict__ v_out,
    int S,
    int H,
    float eps)
{
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int d = threadIdx.x;  // 0..127
    if (h >= H || s >= S || d >= 128) return;

    __shared__ float smem[8];
    const int hd = H * 128;
    const int cdim = 3 * hd;
    const __nv_bfloat16* row = conv_out + (size_t)s * cdim;

    const int q_off = h * 128 + d;
    const int k_off = hd + h * 128 + d;
    const int v_off = 2 * hd + h * 128 + d;
    const int out_idx = (s * H + h) * 128 + d;

    const float qv = bf16_to_f32(row[q_off]);
    const float q_ss = block_reduce_sum(qv * qv, smem);
    q_l2[out_idx] = f32_to_bf16(qv * rsqrtf(q_ss + eps));

    const float kv = bf16_to_f32(row[k_off]);
    const float k_ss = block_reduce_sum(kv * kv, smem);
    k_l2[out_idx] = f32_to_bf16(kv * rsqrtf(k_ss + eps));

    v_out[out_idx] = row[v_off];
}

// ---------------------------------------------------------------------------
// 3b. Fused GDN gating (beta = sigmoid(b), g = neg_A_exp * softplus(a + dt_bias))
//     + per-64-token chunk inclusive cumsum of g!
//     Grid: (H, chunks), Block: 64
// ---------------------------------------------------------------------------
__global__ void gdn_gating_cumsum64_bf16_kernel(
    const __nv_bfloat16* __restrict__ ba_in,
    const float* __restrict__ neg_A_exp,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ beta_out,
    __nv_bfloat16* __restrict__ g_cs_out,
    int S,
    int H,
    int ba_stride,
    int b_col_offset)
{
    const int h = blockIdx.x;
    const int chunk = blockIdx.y;
    const int tid = threadIdx.x;  // 0..63
    const int s = chunk * 64 + tid;

    __shared__ float g_smem[64];
    float g_val = 0.0f;
    if (s < S) {
        const __nv_bfloat16* row = ba_in + (size_t)s * ba_stride + b_col_offset;
        const float b = bf16_to_f32(row[h]);
        const float a = bf16_to_f32(row[H + h]);
        const float beta = 1.0f / (1.0f + __expf(-b));
        beta_out[s * H + h] = f32_to_bf16(beta);
        const float x = a + dt_bias[h];
        // Numerically stable softplus(x)
        const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
        g_val = neg_A_exp[h] * sp;
    }
    g_smem[tid] = g_val;
    __syncthreads();

    // Hillis-Steele inclusive scan over 64 elements in shared memory
    #pragma unroll
    for (int step = 1; step < 64; step <<= 1) {
        float add = (tid >= step) ? g_smem[tid - step] : 0.0f;
        __syncthreads();
        g_smem[tid] += add;
        __syncthreads();
    }
    if (s < S) {
        g_cs_out[s * H + h] = f32_to_bf16(g_smem[tid]);
    }
}

// ---------------------------------------------------------------------------
// 4. Fused Full-Attention QKV split + Q/K RMSNorm + partial RoPE + KV scatter
//    Input: qkv_in [M, nh_q * 2 * hd + nh_kv * hd + nh_kv * hd]
//    Grid: (nh_q + 2 * nh_kv, M), Block: 256 (supports hd <= 256)
// ---------------------------------------------------------------------------
__global__ void qkv_split_norm_rope_kvwrite_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv_in,
    const __nv_bfloat16* __restrict__ q_norm_w1,
    const __nv_bfloat16* __restrict__ k_norm_w1,
    const __nv_bfloat16* __restrict__ cos_tab,
    const __nv_bfloat16* __restrict__ sin_tab,
    const int64_t* __restrict__ dst_idx,  // nullptr => write to row m directly
    __nv_bfloat16* __restrict__ q_out,    // [M, nh_q, hd]
    __nv_bfloat16* __restrict__ kv_k,     // [max_kv_len, nh_kv, hd]
    __nv_bfloat16* __restrict__ kv_v,     // [max_kv_len, nh_kv, hd]
    int M,
    int nh_q,
    int nh_kv,
    int hd,
    int rdim,
    float eps)
{
    const int head_slot = blockIdx.x;
    const int m = blockIdx.y;
    const int d = threadIdx.x;
    if (m >= M || d >= hd) return;

    const int qkv_stride = nh_q * 2 * hd + 2 * nh_kv * hd;
    const __nv_bfloat16* row = qkv_in + (size_t)m * qkv_stride;

    if (head_slot < nh_q) {
        // Q head: RMSNorm + partial RoPE
        __shared__ float smem[8];
        __shared__ float q_smem[256];
        const int h = head_slot;
        const float val = bf16_to_f32(row[h * 2 * hd + d]);
        const float ss = block_reduce_sum(val * val, smem);
        const float inv_rms = rsqrtf(ss / float(hd) + eps);
        const float normed = val * inv_rms * bf16_to_f32(q_norm_w1[d]);
        q_smem[d] = normed;
        __syncthreads();

        float out_v = normed;
        if (d < rdim) {
            const int half = rdim >> 1;
            const float c = bf16_to_f32(cos_tab[m * rdim + d]);
            const float s = bf16_to_f32(sin_tab[m * rdim + d]);
            const float rot = (d < half) ? -q_smem[d + half] : q_smem[d - half];
            out_v = normed * c + rot * s;
        }
        q_out[(size_t)(m * nh_q + h) * hd + d] = f32_to_bf16(out_v);
    } else if (head_slot < nh_q + nh_kv) {
        // K head: RMSNorm + partial RoPE -> write to kv_k[dst_m, h, d]
        __shared__ float smem[8];
        __shared__ float k_smem[256];
        const int h = head_slot - nh_q;
        const int k_base = nh_q * 2 * hd + h * hd;
        const float val = bf16_to_f32(row[k_base + d]);
        const float ss = block_reduce_sum(val * val, smem);
        const float inv_rms = rsqrtf(ss / float(hd) + eps);
        const float normed = val * inv_rms * bf16_to_f32(k_norm_w1[d]);
        k_smem[d] = normed;
        __syncthreads();

        float out_v = normed;
        if (d < rdim) {
            const int half = rdim >> 1;
            const float c = bf16_to_f32(cos_tab[m * rdim + d]);
            const float s = bf16_to_f32(sin_tab[m * rdim + d]);
            const float rot = (d < half) ? -k_smem[d + half] : k_smem[d - half];
            out_v = normed * c + rot * s;
        }
        const int64_t dst_m = dst_idx ? dst_idx[m] : (int64_t)m;
        kv_k[(size_t)(dst_m * nh_kv + h) * hd + d] = f32_to_bf16(out_v);
    } else {
        // V head: direct scatter to kv_v[dst_m, h, d]
        const int h = head_slot - nh_q - nh_kv;
        const int v_base = nh_q * 2 * hd + nh_kv * hd + h * hd;
        const int64_t dst_m = dst_idx ? dst_idx[m] : (int64_t)m;
        kv_v[(size_t)(dst_m * nh_kv + h) * hd + d] = row[v_base + d];
    }
}

// ---------------------------------------------------------------------------
// 5. Fused attn_o * sigmoid(gate) reading gate directly from qkv_in
// ---------------------------------------------------------------------------
__global__ void attn_gate_sigmoid_from_qkv_bf16_kernel(
    const __nv_bfloat16* __restrict__ attn_o,
    const __nv_bfloat16* __restrict__ qkv_in,
    __nv_bfloat16* __restrict__ out,
    int M,
    int nh_q,
    int hd,
    int qkv_stride)
{
    const int total = M * nh_q * hd;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int d = idx % hd;
    const int tmp = idx / hd;
    const int h = tmp % nh_q;
    const int m = tmp / nh_q;
    const float a = bf16_to_f32(attn_o[idx]);
    const float g = bf16_to_f32(qkv_in[(size_t)m * qkv_stride + h * 2 * hd + hd + d]);
    const float sig = 1.0f / (1.0f + __expf(-g));
    out[idx] = f32_to_bf16(a * sig);
}

// ---------------------------------------------------------------------------
// 6a. Fused Residual Add + LayerNorm (for Qwen3.5 ViT blocks)
//     If x != nullptr: residual[m, d] += x[m, d] in-place, then out = LN(residual)
//     If x == nullptr: out = LN(residual)
//     Grid: (M,), Block: 256
// ---------------------------------------------------------------------------
__global__ void residual_add_layernorm_bf16_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int M,
    int dim,
    float eps)
{
    const int m = blockIdx.x;
    if (m >= M) return;
    __shared__ float smem[8];
    __nv_bfloat16* r_row = residual + (size_t)m * dim;
    const __nv_bfloat16* x_row = x ? (x + (size_t)m * dim) : nullptr;
    __nv_bfloat16* o_row = out + (size_t)m * dim;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = bf16_to_f32(r_row[d]);
        if (x_row) {
            v += bf16_to_f32(x_row[d]);
            r_row[d] = f32_to_bf16(v);
        }
        sum += v;
        sq_sum += v * v;
    }
    const float mean = block_reduce_sum(sum, smem) / float(dim);
    const float mean_sq = block_reduce_sum(sq_sum, smem) / float(dim);
    const float inv_std = rsqrtf(fmaxf(mean_sq - mean * mean, 0.0f) + eps);

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = bf16_to_f32(r_row[d]);
        const float n = (v - mean) * inv_std * bf16_to_f32(weight[d]) + bf16_to_f32(bias[d]);
        o_row[d] = f32_to_bf16(n);
    }
}

// ---------------------------------------------------------------------------
// 6b. Fused ViT QKV split + 2D RoPE + transpose [B, S, 3, nh, hd] -> [B, nh, S, hd]
//     Grid: (nh, B * S), Block: 64 (hd == 64)
// ---------------------------------------------------------------------------
__global__ void vit_qkv_rope_transpose_d64_bf16_kernel(
    const __nv_bfloat16* __restrict__ qkv,  // [B, S, 3, nh, 64]
    const __nv_bfloat16* __restrict__ cos_tab,  // [B, S, 64]
    const __nv_bfloat16* __restrict__ sin_tab,  // [B, S, 64]
    __nv_bfloat16* __restrict__ q_out,  // [B, nh, S, 64]
    __nv_bfloat16* __restrict__ k_out,  // [B, nh, S, 64]
    __nv_bfloat16* __restrict__ v_out,  // [B, nh, S, 64]
    int B,
    int S,
    int nh)
{
    const int h = blockIdx.x;
    const int bs = blockIdx.y;
    const int d = threadIdx.x;  // 0..63
    if (h >= nh || bs >= B * S || d >= 64) return;

    const int b = bs / S;
    const int s = bs - b * S;
    __shared__ float q_sm[64];
    __shared__ float k_sm[64];

    const size_t base = (size_t)bs * 3 * nh * 64;
    const float qv = bf16_to_f32(qkv[base + (0 * nh + h) * 64 + d]);
    const float kv = bf16_to_f32(qkv[base + (1 * nh + h) * 64 + d]);
    const __nv_bfloat16 vv = qkv[base + (2 * nh + h) * 64 + d];
    q_sm[d] = qv;
    k_sm[d] = kv;
    __syncthreads();

    const float c = bf16_to_f32(cos_tab[bs * 64 + d]);
    const float sn = bf16_to_f32(sin_tab[bs * 64 + d]);
    const float q_rot = (d < 32) ? -q_sm[d + 32] : q_sm[d - 32];
    const float k_rot = (d < 32) ? -k_sm[d + 32] : k_sm[d - 32];

    const size_t dst = ((size_t)(b * nh + h) * S + s) * 64 + d;
    q_out[dst] = f32_to_bf16(qv * c + q_rot * sn);
    k_out[dst] = f32_to_bf16(kv * c + k_rot * sn);
    v_out[dst] = vv;
}

// ---------------------------------------------------------------------------
// 6c. Fast in-place GELU-tanh (for Qwen3.5 ViT MLP)
// ---------------------------------------------------------------------------
__global__ void gelu_tanh_inplace_bf16_kernel(__nv_bfloat16* __restrict__ x, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = bf16_to_f32(x[idx]);
    // sqrt(2/pi) = 0.7978845608f
    const float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    const float out = 0.5f * v * (1.0f + tanhf(inner));
    x[idx] = f32_to_bf16(out);
}

}  // namespace

// ---- Python C-pointer entrypoints ----

static void silu_mul_split_bf16(uintptr_t gu, uintptr_t out, int M, int inter, uintptr_t stream) {
    int total = M * inter;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    silu_mul_split_bf16_kernel<<<blocks, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(gu),
        reinterpret_cast<__nv_bfloat16*>(out),
        M, inter);
}

static void split_qkvz_bf16(uintptr_t src, uintptr_t qkv_out, uintptr_t z_out,
                            int M, int cdim, int vdim, int stride, uintptr_t stream) {
    int cdim_v8 = cdim >> 3;
    int vdim_v8 = vdim >> 3;
    int stride_v8 = stride >> 3;
    int total_v8 = cdim_v8 + vdim_v8;
    dim3 block(256);
    dim3 grid((total_v8 + 255) / 256, M);
    split_qkvz_bf16_vec8_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const int4*>(src),
        reinterpret_cast<int4*>(qkv_out),
        reinterpret_cast<int4*>(z_out),
        M, cdim_v8, vdim_v8, stride_v8);
}

static void gdn_prep_wy_bf16(
    uintptr_t conv_out, uintptr_t ba_in, uintptr_t neg_A_exp, uintptr_t dt_bias,
    uintptr_t q_l2, uintptr_t k_l2, uintptr_t v_out, uintptr_t beta_out, uintptr_t g_cs_out,
    int S, int H, int D, int ba_stride, int b_col_offset, float eps, uintptr_t stream)
{
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid_a(H, S);
    gdn_split_l2norm_d128_bf16_kernel<<<grid_a, 128, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(conv_out),
        reinterpret_cast<__nv_bfloat16*>(q_l2),
        reinterpret_cast<__nv_bfloat16*>(k_l2),
        reinterpret_cast<__nv_bfloat16*>(v_out),
        S, H, eps);
    int chunks = (S + 63) / 64;
    dim3 grid_b(H, chunks);
    gdn_gating_cumsum64_bf16_kernel<<<grid_b, 64, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(ba_in),
        reinterpret_cast<const float*>(neg_A_exp),
        reinterpret_cast<const float*>(dt_bias),
        reinterpret_cast<__nv_bfloat16*>(beta_out),
        reinterpret_cast<__nv_bfloat16*>(g_cs_out),
        S, H, ba_stride, b_col_offset);
}

static void qkv_split_norm_rope_kvwrite_bf16(
    uintptr_t qkv_in, uintptr_t q_norm_w1, uintptr_t k_norm_w1,
    uintptr_t cos_tab, uintptr_t sin_tab, uintptr_t dst_idx,
    uintptr_t q_out, uintptr_t kv_k, uintptr_t kv_v,
    int M, int nh_q, int nh_kv, int hd, int rdim, float eps, uintptr_t stream)
{
    dim3 grid(nh_q + 2 * nh_kv, M);
    qkv_split_norm_rope_kvwrite_bf16_kernel<<<grid, hd, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv_in),
        reinterpret_cast<const __nv_bfloat16*>(q_norm_w1),
        reinterpret_cast<const __nv_bfloat16*>(k_norm_w1),
        reinterpret_cast<const __nv_bfloat16*>(cos_tab),
        reinterpret_cast<const __nv_bfloat16*>(sin_tab),
        reinterpret_cast<const int64_t*>(dst_idx),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(kv_k),
        reinterpret_cast<__nv_bfloat16*>(kv_v),
        M, nh_q, nh_kv, hd, rdim, eps);
}

static void attn_gate_sigmoid_from_qkv_bf16(
    uintptr_t attn_o, uintptr_t qkv_in, uintptr_t out,
    int M, int nh_q, int hd, int qkv_stride, uintptr_t stream)
{
    int total = M * nh_q * hd;
    int blocks = (total + 255) / 256;
    attn_gate_sigmoid_from_qkv_bf16_kernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(attn_o),
        reinterpret_cast<const __nv_bfloat16*>(qkv_in),
        reinterpret_cast<__nv_bfloat16*>(out),
        M, nh_q, hd, qkv_stride);
}

static void residual_add_layernorm_bf16(
    uintptr_t residual, uintptr_t x, uintptr_t weight, uintptr_t bias, uintptr_t out,
    int M, int dim, float eps, uintptr_t stream)
{
    residual_add_layernorm_bf16_kernel<<<M, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out),
        M, dim, eps);
}

static void vit_qkv_rope_transpose_d64_bf16(
    uintptr_t qkv, uintptr_t cos_tab, uintptr_t sin_tab,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int B, int S, int nh, uintptr_t stream)
{
    dim3 grid(nh, B * S);
    vit_qkv_rope_transpose_d64_bf16_kernel<<<grid, 64, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(cos_tab),
        reinterpret_cast<const __nv_bfloat16*>(sin_tab),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<__nv_bfloat16*>(v_out),
        B, S, nh);
}

static void gelu_tanh_inplace_bf16(uintptr_t x, int n, uintptr_t stream) {
    int blocks = (n + 255) / 256;
    gelu_tanh_inplace_bf16_kernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<__nv_bfloat16*>(x), n);
}

PYBIND11_MODULE(internvla_orin_ops, m) {
    m.doc() = "Fused SM87 (Jetson Orin) BF16 CUDA kernels for InternVLA-A1.5 FlashRT backend";
    m.def("silu_mul_split_bf16", &silu_mul_split_bf16);
    m.def("split_qkvz_bf16", &split_qkvz_bf16);
    m.def("gdn_prep_wy_bf16", &gdn_prep_wy_bf16);
    m.def("qkv_split_norm_rope_kvwrite_bf16", &qkv_split_norm_rope_kvwrite_bf16);
    m.def("attn_gate_sigmoid_from_qkv_bf16", &attn_gate_sigmoid_from_qkv_bf16);
    m.def("residual_add_layernorm_bf16", &residual_add_layernorm_bf16);
    m.def("vit_qkv_rope_transpose_d64_bf16", &vit_qkv_rope_transpose_d64_bf16);
    m.def("gelu_tanh_inplace_bf16", &gelu_tanh_inplace_bf16);
}
