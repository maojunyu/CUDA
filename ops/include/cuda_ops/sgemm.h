#pragma once

#include "cuda_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// 行主序 GEMM：C[M×N] = A[M×K] × B[K×N]，float32。
//
// d_A / d_B / d_C 均为设备指针。
// stream 传 cudaStream_t（强转成 void*），传 0 或 nullptr 表示默认流。
//
// 内部策略：
//   * M、N 为 128 的倍数且 K 为 8 的倍数时 → 自研最优 kernel
//     （V7 双缓冲 + 转置 A + float4，sm_120 上 ~28 TFLOPS）；
//   * 其余尺寸 → 回退 cuBLAS（保证任意尺寸正确）。
//
// 返回 0 表示成功，否则为 cudaError_t 错误码。
CUDA_OPS_API int cuda_ops_sgemm(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K,
    void* stream);

#ifdef __cplusplus
}
#endif
