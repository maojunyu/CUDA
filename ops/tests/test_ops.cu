// cuda_ops 库冒烟测试：链接 DLL，用 CPU 参考实现校验 SGEMM 与 FlashAttention。
// 用法：构建后运行 build/<preset>/ops/tests/<config>/test_ops.exe

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_ops/cuda_ops.h"
#include "cuda_ops/sgemm.h"
#include "cuda_ops/flash_attention.h"

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err_ = (expr);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

static void fill_random(std::vector<float>& v, unsigned seed) {
    std::srand(seed);
    for (float& x : v) {
        x = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
}

// ---------- SGEMM ----------
static void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

static bool test_sgemm() {
    const int M = 256, N = 256, K = 256;   // 对齐 128/8，命中 V7 快速路径
    std::vector<float> hA(M * K), hB(K * N), hC(M * N), hRef(M * N);
    fill_random(hA, 1u);
    fill_random(hB, 2u);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int st = cuda_ops_sgemm(dA, dB, dC, M, N, K, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_sgemm(hA.data(), hB.data(), hRef.data(), M, N, K);

    float max_err = 0.0f;
    for (size_t i = 0; i < hC.size(); ++i) {
        max_err = std::fmax(max_err, std::fabs(hC[i] - hRef[i]));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    const bool pass = (st == 0) && (max_err < 1e-3f);
    std::printf("SGEMM          status=%d  max_err=%.6g  -> %s\n",
                st, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ---------- FlashAttention ----------
static constexpr int kHeadDim = 64;

static void cpu_attention(const float* Q, const float* K, const float* V, float* O,
                          int bh_count, int seq_len, float scale, bool causal) {
    const size_t head = (size_t)seq_len * kHeadDim;
    for (int bh = 0; bh < bh_count; ++bh) {
        const float* qh = Q + bh * head;
        const float* kh = K + bh * head;
        const float* vh = V + bh * head;
        float* oh = O + bh * head;
        for (int i = 0; i < seq_len; ++i) {
            const int j_end = causal ? (i + 1) : seq_len;
            double m = -INFINITY;
            for (int j = 0; j < j_end; ++j) {
                double s = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    s += (double)qh[i * kHeadDim + d] * (double)kh[j * kHeadDim + d];
                }
                s *= scale;
                m = std::max(m, s);
            }
            double l = 0.0;
            for (int j = 0; j < j_end; ++j) {
                double s = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    s += (double)qh[i * kHeadDim + d] * (double)kh[j * kHeadDim + d];
                }
                l += std::exp(s * scale - m);
            }
            for (int d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (int j = 0; j < j_end; ++j) {
                    double s = 0.0;
                    for (int dd = 0; dd < kHeadDim; ++dd) {
                        s += (double)qh[i * kHeadDim + dd] * (double)kh[j * kHeadDim + dd];
                    }
                    acc += std::exp(s * scale - m) * (double)vh[j * kHeadDim + d];
                }
                oh[i * kHeadDim + d] = (float)(acc / l);
            }
        }
    }
}

static bool test_flash_attention() {
    const int batch = 1, heads = 1, seq_len = 128;
    const int bh_count = batch * heads;
    const float scale = 1.0f / std::sqrt((float)kHeadDim);
    const size_t elems = (size_t)bh_count * seq_len * kHeadDim;

    std::vector<float> hQ(elems), hK(elems), hV(elems), hO(elems), hRef(elems);
    fill_random(hQ, 11u);
    fill_random(hK, 22u);
    fill_random(hV, 33u);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    CUDA_CHECK(cudaMalloc(&dQ, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

    bool ok = true;
    for (const bool causal : {false, true}) {
        CUDA_CHECK(cudaMemset(dO, 0, elems * sizeof(float)));
        const int st = cuda_ops_flash_attention_fwd(dQ, dK, dV, dO, batch, heads,
                                                    seq_len, kHeadDim, scale, causal, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hO.data(), dO, elems * sizeof(float), cudaMemcpyDeviceToHost));

        cpu_attention(hQ.data(), hK.data(), hV.data(), hRef.data(), bh_count, seq_len,
                      scale, causal);

        float max_err = 0.0f;
        for (size_t i = 0; i < hO.size(); ++i) {
            max_err = std::fmax(max_err, std::fabs(hO[i] - hRef[i]));
        }
        const bool pass = (st == 0) && (max_err < 2e-3f);
        ok = ok && pass;
        std::printf("FlashAttention (%s) status=%d  max_err=%.6g  -> %s\n",
                    causal ? "causal    " : "non-causal", st, max_err, pass ? "PASS" : "FAIL");
    }

    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dK));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dO));
    return ok;
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("cuda_ops version: %s\n\n", cuda_ops_version());

    const bool ok_sgemm = test_sgemm();
    const bool ok_fa = test_flash_attention();

    const bool ok = ok_sgemm && ok_fa;
    std::printf("\n%s\n", ok ? "cuda_ops DLL OK!" : "cuda_ops DLL FAILED!");
    return ok ? 0 : 1;
}
