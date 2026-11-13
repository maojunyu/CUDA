// FlashAttention 前向（分块 + 在线 softmax）手写 CUDA 实现，并与 CPU 参考结果比对。
// 数据布局：[batch, head, seq_len, head_dim]，行主序，float32。
// 本示例固定 head_dim = 64，查询分块 Br = 64，键分块 Bc = 32，4 个 warp。
//
// 算法要点（FlashAttention v1 思路）：
//   * 每个 block 只负责 Br 行查询，K/V 按 Bc 列分块，从头流式扫描一遍；
//   * 用 running max(m) 与 running sum(l) 做在线 softmax，避免 N x N 的注意力矩阵落显存；
//   * O 累加器保存在寄存器中，每换一个 K/V 分块先按 alpha = exp(m_old - m_new) 重缩放。

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err_ = (expr);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,  \
                         cudaGetErrorString(err_));                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

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

// CPU 参考实现：朴素的 softmax(QK^T / sqrt(D)) V，用 double 计算以减少误差
void attention_reference(const float* Q, const float* K, const float* V, float* O,
                         int bh_count, int seq_len, float scale, bool causal) {
    const size_t head_offset = (size_t)seq_len * kHeadDim;
    std::vector<double> scores(seq_len);

    for (int bh = 0; bh < bh_count; ++bh) {
        const float* qh = Q + bh * head_offset;
        const float* kh = K + bh * head_offset;
        const float* vh = V + bh * head_offset;
        float* oh = O + bh * head_offset;

        for (int i = 0; i < seq_len; ++i) {
            const int j_end = causal ? (i + 1) : seq_len;
            double m = -INFINITY;
            for (int j = 0; j < j_end; ++j) {
                double s = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    s += static_cast<double>(qh[i * kHeadDim + d]) *
                         static_cast<double>(kh[j * kHeadDim + d]);
                }
                s *= scale;
                scores[j] = s;
                m = std::max(m, s);
            }

            double l = 0.0;
            for (int j = 0; j < j_end; ++j) {
                scores[j] = std::exp(scores[j] - m);
                l += scores[j];
            }

            for (int d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (int j = 0; j < j_end; ++j) {
                    acc += scores[j] * static_cast<double>(vh[j * kHeadDim + d]);
                }
                oh[i * kHeadDim + d] = static_cast<float>(acc / l);
            }
        }
    }
}

void fill_random(std::vector<float>& v, unsigned seed) {
    std::srand(seed);
    for (float& x : v) {
        x = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        err = std::fmax(err, std::fabs(a[i] - b[i]));
    }
    return err;
}

}  // namespace

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);

    const int batch = 2;
    const int heads = 2;
    const int seq_len = 512;
    const int bh_count = batch * heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    const size_t elems = (size_t)bh_count * seq_len * kHeadDim;
    const size_t bytes = elems * sizeof(float);

    std::vector<float> h_q(elems), h_k(elems), h_v(elems);
    std::vector<float> h_o(elems, 0.0f), h_ref(elems, 0.0f);
    fill_random(h_q, 2024u);
    fill_random(h_k, 7u);
    fill_random(h_v, 13u);

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_k, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_o, bytes));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(kThreads);
    const dim3 grid((seq_len + kBr - 1) / kBr, bh_count);

    bool ok = true;
    for (const bool causal : {false, true}) {
        CUDA_CHECK(cudaMemset(d_o, 0, bytes));
        flash_attention_fwd<<<grid, block>>>(d_q, d_k, d_v, d_o, seq_len, scale, causal);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, bytes, cudaMemcpyDeviceToHost));

        attention_reference(h_q.data(), h_k.data(), h_v.data(), h_ref.data(), bh_count,
                            seq_len, scale, causal);

        const float err = max_abs_diff(h_o, h_ref);
        const bool pass = err < 2e-3f;
        ok = ok && pass;
        std::printf("FlashAttention (%s) max error: %.6g  -> %s\n",
                    causal ? "causal    " : "non-causal", err, pass ? "PASS" : "FAIL");
    }

    // ---- 计时（因果模式，取平均）----
    constexpr int kIters = 50;
    cudaEvent_t ev_start{}, ev_stop{};
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int it = 0; it < kIters; ++it) {
        flash_attention_fwd<<<grid, block>>>(d_q, d_k, d_v, d_o, seq_len, scale, true);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    const float avg_ms = total_ms / kIters;
    const double flops = 2.0 * 2.0 * bh_count * (double)seq_len * seq_len * kHeadDim * 0.5;
    std::printf("causal kernel: %.3f ms / iter (%.1f GFLOP/s)\n", avg_ms,
                flops / (avg_ms * 1e6));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));

    std::printf("%s\n", ok ? "FlashAttention CUDA OK!" : "FlashAttention CUDA FAILED!");
    return ok ? 0 : 1;
}
