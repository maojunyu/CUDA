// cuBLAS GEMM：行主序 C = alpha * A(MxK) * B(KxN) + beta * C
// 使用 cuBLAS 的 cublasSgemm（FP32）与 cublasGemmEx（TF32）两种路径，
// 与 CPU 朴素版结果比对，并输出耗时与 GFLOP/s。
//
// 注意：cuBLAS 只有列主序接口。行主序的 A(MxK)、B(KxN) 从列主序看就是 A^T(KxM)、B^T(NxK)，
// 因此计算 C(MxN) 等价于列主序下算 C^T(NxM) = B^T(NxK) * A^T(KxM)，
// 即把 B、A 的传参顺序对调，且 leading dimension 分别为 N 和 K。

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
constexpr int REPEAT = 5;   // 计时重复次数，取平均值

bool checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        printf("[CUDA ERROR]   %s: %s\n", what, cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool checkCublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR] %s: code = %d\n", what, static_cast<int>(status));
        return false;
    }
    return true;
}

// 设备端计时（CUDA Event，单位 ms）
struct CudaStepTimer {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CudaStepTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaStepTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin() { cudaEventRecord(start); }

    float end(const char* name) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  [GPU] %-32s %9.3f ms\n", name, ms);
        return ms;
    }
};

// 主机端计时（std::chrono，单位 ms）
struct HostStepTimer {
    std::chrono::high_resolution_clock::time_point t0;

    void begin() { t0 = std::chrono::high_resolution_clock::now(); }

    double end(const char* name) {
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  [CPU] %-32s %9.3f ms\n", name, ms);
        return ms;
    }
};

void randomMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// 以 C_ref 为基准比对，输出最大绝对/相对误差与不匹配元素个数
void compareMatrix(const float* C_ref, const float* C_test, int rows, int cols,
                   double tolerance) {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    int mismatches = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const int idx = i * cols + j;
            const double ref = static_cast<double>(C_ref[idx]);
            const double diff = std::fabs(ref - static_cast<double>(C_test[idx]));
            const double rel = diff / std::max(1.0, std::fabs(ref));

            max_abs_diff = std::max(max_abs_diff, diff);
            max_rel_diff = std::max(max_rel_diff, rel);

            if (rel > tolerance) {
                if (mismatches < 5) {
                    printf("  Mismatch at (%d, %d): CPU = %f, GPU = %f, rel = %.3e\n",
                           i, j, C_ref[idx], C_test[idx], rel);
                }
                ++mismatches;
            }
        }
    }

    printf("[Compare] max_abs_diff = %.6e, max_rel_diff = %.6e, mismatches = %d / %d\n",
           max_abs_diff, max_rel_diff, mismatches, rows * cols);
    printf("[Compare] %s (tolerance: rel <= %.0e)\n",
           mismatches == 0 ? "PASS - CPU and GPU results match."
                           : "FAIL - CPU and GPU results differ.",
           tolerance);
}

// CPU 朴素版行主序 SGEMM，作为结果基准
void cpuSgemm(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                sum += A[i * k + t] * B[t * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// 行主序 GEMM 的 cuBLAS 封装：C(MxN) = A(MxK) * B(KxN)
bool cublasSgemmRowMajor(cublasHandle_t handle, const float* dA, const float* dB,
                         float* dC) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 列主序下：C^T(NxM) = B^T(NxK) * A^T(KxM)
    // m = N, n = M, k = K；lda = N（B 的列主序行数）、ldb = K（A 的列主序行数）、ldc = N
    const cublasStatus_t status = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB, N,
        dA, K,
        &beta,
        dC, N);

    return checkCublas(status, "cublasSgemm");
}

// 同上，但用 TF32 Tensor Core 计算（输入仍是 FP32，内部降为 10 位尾数）
bool cublasGemmExTf32(cublasHandle_t handle, const float* dA, const float* dB,
                      float* dC) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const cublasStatus_t status = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB, CUDA_R_32F, N,
        dA, CUDA_R_32F, K,
        &beta,
        dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    return checkCublas(status, "cublasGemmEx(TF32)");
}

double gflops(double ms) { return 2.0 * M * N * K / (ms * 1e6); }

}  // namespace

int main() {
    // 0. 打印设备信息
    int device = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("GEMM: M = %d, N = %d, K = %d, repeat = %d\n\n", M, N, K, REPEAT);

    HostStepTimer host_timer;
    CudaStepTimer gpu_timer;

    // 1. 主机内存 + 初始化
    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    float* h_A = static_cast<float*>(malloc(bytesA));
    float* h_B = static_cast<float*>(malloc(bytesB));
    float* h_C_cpu = static_cast<float*>(malloc(bytesC));
    float* h_C_gpu = static_cast<float*>(malloc(bytesC));
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        printf("Host malloc failed.\n");
        return -1;
    }

    host_timer.begin();
    srand(0);
    randomMatrix(h_A, M, K);
    randomMatrix(h_B, K, N);
    host_timer.end("init matrices (rand)");

    // 2. CPU 基准
    printf("\n");
    host_timer.begin();
    cpuSgemm(h_A, h_B, h_C_cpu, M, N, K);
    const double cpu_ms = host_timer.end("cpuSgemm");

    // 3. 设备内存 + H2D
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    if (!checkCuda(cudaMalloc(&d_A, bytesA), "cudaMalloc A") ||
        !checkCuda(cudaMalloc(&d_B, bytesB), "cudaMalloc B") ||
        !checkCuda(cudaMalloc(&d_C, bytesC), "cudaMalloc C")) {
        return -1;
    }

    printf("\n");
    gpu_timer.begin();
    checkCuda(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice), "H2D B");
    gpu_timer.end("cudaMemcpy H2D (A, B)");

    // 4. 创建 cuBLAS handle
    cublasHandle_t handle = nullptr;
    if (!checkCublas(cublasCreate(&handle), "cublasCreate")) {
        return -1;
    }
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);  // FP32 精确计算

    // 5. FP32 SGEMM：先预热，再计时
    printf("\n[cublasSgemm - FP32]\n");
    if (!cublasSgemmRowMajor(handle, d_A, d_B, d_C)) {
        return -1;
    }
    if (!checkCuda(cudaDeviceSynchronize(), "sync after warmup")) {
        return -1;
    }

    gpu_timer.begin();
    for (int r = 0; r < REPEAT; ++r) {
        if (!cublasSgemmRowMajor(handle, d_A, d_B, d_C)) {
            return -1;
        }
    }
    const float sgemm_ms = gpu_timer.end("cublasSgemm x REPEAT") / REPEAT;
    printf("  [GPU] %-32s %9.3f ms  (%.2f GFLOP/s)\n", "average per call",
           sgemm_ms, gflops(sgemm_ms));

    gpu_timer.begin();
    checkCuda(cudaMemcpy(h_C_gpu, d_C, bytesC, cudaMemcpyDeviceToHost), "D2H C");
    gpu_timer.end("cudaMemcpy D2H (C)");

    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N, 1e-4);
    host_timer.end("compareMatrix");

    printf("\n[cublasSgemm Summary] CPU: %.3f ms (%.2f GFLOP/s), cuBLAS: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, gflops(cpu_ms),
           sgemm_ms, gflops(sgemm_ms),
           cpu_ms / sgemm_ms);

    // 6. TF32 Tensor Core 版本（精度较低，容差放宽）
    printf("\n[cublasGemmEx - TF32 Tensor Core]\n");
    cudaMemset(d_C, 0, bytesC);
    if (!cublasGemmExTf32(handle, d_A, d_B, d_C)) {
        return -1;
    }
    if (!checkCuda(cudaDeviceSynchronize(), "sync after warmup tf32")) {
        return -1;
    }

    gpu_timer.begin();
    for (int r = 0; r < REPEAT; ++r) {
        if (!cublasGemmExTf32(handle, d_A, d_B, d_C)) {
            return -1;
        }
    }
    const float tf32_ms = gpu_timer.end("cublasGemmEx(TF32) x REPEAT") / REPEAT;
    printf("  [GPU] %-32s %9.3f ms  (%.2f GFLOP/s)\n", "average per call",
           tf32_ms, gflops(tf32_ms));

    gpu_timer.begin();
    checkCuda(cudaMemcpy(h_C_gpu, d_C, bytesC, cudaMemcpyDeviceToHost), "D2H C (tf32)");
    gpu_timer.end("cudaMemcpy D2H (C)");

    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N, 1e-2);
    host_timer.end("compareMatrix");

    printf("\n[cublasGemmEx(TF32) Summary] cuBLAS FP32: %.3f ms (%.2f GFLOP/s), TF32: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           sgemm_ms, gflops(sgemm_ms),
           tf32_ms, gflops(tf32_ms),
           sgemm_ms / tf32_ms);

    // 7. 释放资源
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}