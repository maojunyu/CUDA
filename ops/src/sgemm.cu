// 算子优化库 —— SGEMM（最优版本）。
// 当前采用 study/sgemm/kernel.cu 中的 V7（双缓冲 + 转置 A + float4 + 寄存器预取），
// 在 sm_120 (RTX 5070 Ti) 上约 28 TFLOPS（FP32 CUDA Core 路线的最终赢家）。
//
// 学习过程中若跑出更优版本，把它替换到这里并更新下方注释即可，对外 API 保持不变。

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_ops/sgemm.h"

// ============ V7: 双缓冲（寄存器预取，BK=8 专用）============
// 要求：M、N 为 BM=128 的倍数，K 为 BK=8 的倍数。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v7(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    const int threadRow = ty * TM;
    const int threadCol = tx * TN;

    const float* APtr = A + by * BM * K;
    const float* BPtr = B + bx * BN;
    float* CPtr = C + by * BM * N + bx * BN;

    float acc[TM][TN] = {{0.0f}};

    const int li = ty * blockDim.x + tx;
    const int aRow = li / (BK / 4);
    const int aCol = (li % (BK / 4)) * 4;
    const int bRow = li / (BN / 4);
    const int bCol = (li % (BN / 4)) * 4;

    {
        float4 vA = *reinterpret_cast<const float4*>(&APtr[aRow * K + aCol]);
        float4 vB = *reinterpret_cast<const float4*>(&BPtr[bRow * N + bCol]);
        As[0][aCol + 0][aRow] = vA.x;  As[0][aCol + 1][aRow] = vA.y;
        As[0][aCol + 2][aRow] = vA.z;  As[0][aCol + 3][aRow] = vA.w;
        Bs[0][bRow][bCol + 0] = vB.x;  Bs[0][bRow][bCol + 1] = vB.y;
        Bs[0][bRow][bCol + 2] = vB.z;  Bs[0][bRow][bCol + 3] = vB.w;
    }
    __syncthreads();

    int buf = 0;
    for (int kb = 0; kb < K; kb += BK) {
        float4 vA_next, vB_next;
        if (kb + BK < K) {
            vA_next = *reinterpret_cast<const float4*>(&APtr[aRow * K + (kb + BK) + aCol]);
            vB_next = *reinterpret_cast<const float4*>(&BPtr[(kb + BK + bRow) * N + bCol]);
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float4 a0 = *reinterpret_cast<const float4*>(&As[buf][k][threadRow]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[buf][k][threadRow + 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][k][threadCol]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][k][threadCol + 4]);
            float a[TM] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[TN] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a[m] * b[n];
        }

        if (kb + BK < K) {
            As[1 - buf][aCol + 0][aRow] = vA_next.x;  As[1 - buf][aCol + 1][aRow] = vA_next.y;
            As[1 - buf][aCol + 2][aRow] = vA_next.z;  As[1 - buf][aCol + 3][aRow] = vA_next.w;
            Bs[1 - buf][bRow][bCol + 0] = vB_next.x;  Bs[1 - buf][bRow][bCol + 1] = vB_next.y;
            Bs[1 - buf][bRow][bCol + 2] = vB_next.z;  Bs[1 - buf][bRow][bCol + 3] = vB_next.w;
        }
        __syncthreads();
        buf = 1 - buf;
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            CPtr[(threadRow + m) * N + threadCol + n] = acc[m][n];
}

namespace {

// 行主序 GEMM 的 cuBLAS 回退：C(M×N) = A(M×K) × B(K×N)。
// cuBLAS 只有列主序接口：行主序 C = A×B 等价于列主序 C^T = B^T × A^T，
// 因此 m=N、n=M、k=K，传参顺序对调：dB(ld=N) × dA(ld=K)。
cublasStatus_t sgemm_cublas_fallback(cublasHandle_t handle,
                                     const float* dA, const float* dB, float* dC,
                                     int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
}

// 懒加载的 cuBLAS handle（首次调用创建，进程内复用，不销毁）。
cublasHandle_t get_handle() {
    static cublasHandle_t h = nullptr;
    if (h == nullptr) {
        cublasCreate(&h);
    }
    return h;
}

}  // namespace

const char* cuda_ops_version(void) {
    return "0.1.0";
}

int cuda_ops_sgemm(const float* d_A, const float* d_B, float* d_C,
                   int M, int N, int K, void* stream) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    if (M % BM == 0 && N % BN == 0 && K % BK == 0) {
        dim3 block(BN / TN, BM / TM);   // 16 × 16 = 256 线程
        dim3 grid(N / BN, M / BM);
        sgemm_v7<BM, BN, BK, TM, TN><<<grid, block, 0, s>>>(d_A, d_B, d_C, M, N, K);
        return static_cast<int>(cudaGetLastError());
    }

    cublasHandle_t h = get_handle();
    if (h == nullptr) {
        return static_cast<int>(cudaErrorUnknown);
    }
    const cublasStatus_t st = sgemm_cublas_fallback(h, d_A, d_B, d_C, M, N, K);
    return (st == CUBLAS_STATUS_SUCCESS) ? 0 : static_cast<int>(cudaErrorUnknown);
}
