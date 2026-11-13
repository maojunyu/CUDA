// 算子优化库 —— FlashAttention 前向（最优版本）。
// 采用 study/flash_attention/kernel.cu 中的实现：分块 + 在线 softmax（FlashAttention v1 思路）。
// 当前固定 head_dim = 64，查询分块 Br = 64，键分块 Bc = 32，4 个 warp。
//
// 学习过程中若跑出更优版本（如加入 Tensor Core / TMA / cluster），
// 把它替换到这里即可，对外 API 保持不变。

#include <cuda_runtime.h>
#include <cmath>

#include "cuda_ops/flash_attention.h"

namespace {

constexpr int kHeadDim = 64;                          // 每个注意力头的维度 D
constexpr int kBr = 64;                               // 查询分块大小
constexpr int kBc = 32;                               // 键/值分块大小
constexpr int kThreads = 128;                         // 每 block 线程数（4 个 warp）
constexpr int kWarps = kThreads / 32;
constexpr int kOElems = kBr * kHeadDim / kThreads;    // 每个线程持有的 O 元素数

// 静态共享内存约 40.75 KB，低于 48 KB 上限，无需 cudaFuncSetAttribute 申请 opt-in。
struct SharedStorage {
    float Qs[kBr * kHeadDim];   // Q 分块
    float Ks[kBc * kHeadDim];   // K 分块
    float Vs[kBc * kHeadDim];   // V 分块
    float S[kBr * kBc];         // 打分矩阵，后续原地复用为 P
    float m_run[kBr];           // 跨 K/V 分块的 running max
    float alpha[kBr];           // 本分块的 O 重缩放系数
    float l_run[kBr];           // 跨 K/V 分块的 running sum
};

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

__global__ __launch_bounds__(kThreads) void flash_attention_fwd(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int seq_len, float softmax_scale, bool causal) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    const int q_start = blockIdx.x * kBr;                       // 本 block 的查询起点
    const size_t head_offset = (size_t)blockIdx.y * seq_len * kHeadDim;

    Q += head_offset;
    K += head_offset;
    V += head_offset;
    O += head_offset;

    __shared__ SharedStorage sm;

    // ---- 载入 Q 分块（越界的查询行填 0，结果最终会被丢弃）----
    for (int idx = tid; idx < kBr * kHeadDim; idx += kThreads) {
        const int r = idx / kHeadDim;
        const int c = idx % kHeadDim;
        const int gr = q_start + r;
        sm.Qs[idx] = (gr < seq_len) ? Q[(size_t)gr * kHeadDim + c] : 0.0f;
    }
    if (tid < kBr) {
        sm.m_run[tid] = -INFINITY;
        sm.l_run[tid] = 0.0f;
        sm.alpha[tid] = 0.0f;
    }
    __syncthreads();

    // ---- 寄存器中的输出累加器 ----
    float o_acc[kOElems];
#pragma unroll
    for (int k = 0; k < kOElems; ++k) {
        o_acc[k] = 0.0f;
    }

    // 因果注意力只需扫描到本查询分块的最后一行为止
    const int kv_limit = causal ? min(seq_len, q_start + kBr) : seq_len;

    for (int k_start = 0; k_start < kv_limit; k_start += kBc) {
        // ---- 载入 K/V 分块 ----
        for (int idx = tid; idx < kBc * kHeadDim; idx += kThreads) {
            const int r = idx / kHeadDim;
            const int c = idx % kHeadDim;
            const int gr = k_start + r;
            if (gr < seq_len) {
                sm.Ks[idx] = K[(size_t)gr * kHeadDim + c];
                sm.Vs[idx] = V[(size_t)gr * kHeadDim + c];
            } else {
                sm.Ks[idx] = 0.0f;
                sm.Vs[idx] = 0.0f;
            }
        }
        __syncthreads();

        // ---- S = Q * K^T * scale，每个线程负责 kOElems 个元素 ----
        for (int idx = tid; idx < kBr * kBc; idx += kThreads) {
            const int i = idx / kBc;
            const int j = idx % kBc;
            const int qi = q_start + i;
            const int kj = k_start + j;
            const float* q_row = sm.Qs + i * kHeadDim;
            const float* k_row = sm.Ks + j * kHeadDim;

            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                s += q_row[d] * k_row[d];
            }
            s *= softmax_scale;

            // 越界与因果掩码；越界查询行按最后一行掩码，避免整行 -inf 产生 NaN
            const int qi_mask = (qi < seq_len) ? qi : (seq_len - 1);
            if (kj >= seq_len || (causal && kj > qi_mask)) {
                s = -INFINITY;
            }
            sm.S[idx] = s;
        }
        __syncthreads();

        // ---- 行最大值 + 在线 softmax 状态更新（一个 warp 负责若干行）----
        for (int i = warp; i < kBr; i += kWarps) {
            const float mx = warp_max(sm.S[i * kBc + lane]);
            if (lane == 0) {
                const float m_prev = sm.m_run[i];
                const float m_new = fmaxf(m_prev, mx);
                sm.alpha[i] = (m_prev == -INFINITY) ? 0.0f : __expf(m_prev - m_new);
                sm.m_run[i] = m_new;
            }
        }
        __syncthreads();

        // ---- P = exp(S - m_new) 原地写回，并更新 running sum ----
        for (int i = warp; i < kBr; i += kWarps) {
            const float m = sm.m_run[i];
            const float p = __expf(sm.S[i * kBc + lane] - m);
            sm.S[i * kBc + lane] = p;
            const float row_sum = warp_sum(p);
            if (lane == 0) {
                sm.l_run[i] = sm.l_run[i] * sm.alpha[i] + row_sum;
            }
        }
        __syncthreads();

        // ---- O 重缩放并累加 P * V ----
#pragma unroll
        for (int k = 0; k < kOElems; ++k) {
            const int idx = tid + k * kThreads;
            const int i = idx / kHeadDim;   // 输出行
            const int x = idx % kHeadDim;   // 输出列
            const float* p_row = sm.S + i * kBc;
            const float* v_col = sm.Vs + x;

            float acc = o_acc[k] * sm.alpha[i];
#pragma unroll 4
            for (int j = 0; j < kBc; ++j) {
                acc += p_row[j] * v_col[j * kHeadDim];
            }
            o_acc[k] = acc;
        }
        __syncthreads();
    }

    // ---- 写回 O = O_acc / l ----
#pragma unroll
    for (int k = 0; k < kOElems; ++k) {
        const int idx = tid + k * kThreads;
        const int i = idx / kHeadDim;
        const int x = idx % kHeadDim;
        const int gr = q_start + i;
        if (gr < seq_len) {
            O[(size_t)gr * kHeadDim + x] = o_acc[k] / sm.l_run[i];
        }
    }
}

}  // namespace

int cuda_ops_flash_attention_fwd(const float* d_Q, const float* d_K,
                                 const float* d_V, float* d_O,
                                 int batch, int heads, int seq_len, int head_dim,
                                 float softmax_scale, int causal, void* stream) {
    if (head_dim != kHeadDim) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const int bh_count = batch * heads;
    const dim3 block(kThreads);
    const dim3 grid((seq_len + kBr - 1) / kBr, bh_count);

    flash_attention_fwd<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        d_Q, d_K, d_V, d_O, seq_len, softmax_scale, causal != 0);

    return static_cast<int>(cudaGetLastError());
}
