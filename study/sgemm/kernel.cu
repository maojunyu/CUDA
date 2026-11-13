// SGEMM：GPU（朴素版）与 CPU（朴素版）矩阵乘法的分步计时 + 结果比对。
// 每一步（malloc、H2D 拷贝、kernel、D2H 拷贝、free）都输出耗时，
// 最后执行 CPU 版矩阵乘法并计时，再以 CPU 结果为基准与 GPU 结果比对。

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
using namespace nvcuda;   // wmma 接口在 nvcuda 命名空间
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

const int M = 4096;
const int N = 4096;
const int K = 4096;
const int BLOCK = 16;

// 设备端分步计时（CUDA Event，单位 ms）
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

// 主机端分步计时（std::chrono，单位 ms）
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
// tol 为相对误差容差；Tensor Core(TF32) 精度 ~1e-3，需传更大的 tol。
void compareMatrix(const float* C_ref, const float* C_test, int rows, int cols, double tol = 1e-4) {
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

            if (rel > tol) {
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
    printf("[Compare] %s (tolerance: rel <= %g)\n",
           mismatches == 0 ? "PASS - CPU and GPU results match."
                           : "FAIL - CPU and GPU results differ.", tol);
}

void cpuSgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 行主序 GEMM 的 cuBLAS 封装：C(MxN) = A(MxK) * B(KxN)
// cuBLAS 只有列主序接口，行主序 C = A*B 等价于列主序 C^T = B^T * A^T，
// 因此传参顺序对调：m=N, n=M, k=K；lda=N, ldb=K, ldc=N。
void cublasSgemmRowMajor(cublasHandle_t handle, const float* dA, const float* dB, float* dC) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
}
//初始版本 每个线程从全局内存中读取一行和一列
//读：MN(K + K)
//写：MN
__global__ void cudaSgemm_V0(float* A, float* B, float* C, int M, int N, int K) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < M && col < N) {
        float* APtr = A + blockIdx.y * blockDim.y * K;
        float* BPtr = B + blockIdx.x * blockDim.x;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += APtr[threadIdx.y * K + k] * BPtr[k * N + threadIdx.x];
        }
        C[row * N + col] = sum;
    }
}

//使用共享内存 将A的一整行和B的一整列存入到sharedMemory中
//存在的问题：共享内存需要的太大，不适合大矩阵的乘法
template<unsigned int BLOCK_,unsigned int K_>
__global__ void cudaSgemm_V1(float* A,float* B, float* C, int M, int N, int K) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float AShared[BLOCK_][K_];
    __shared__ float BShared[K_][BLOCK_];
    float* APtr = A + blockIdx.y * blockDim.y * K;
    float* BPtr = B + blockIdx.x * blockDim.x;
    if (row < M && col < N) {
        for (int s = 0; s < K; s += blockDim.x) {
            AShared[threadIdx.y][threadIdx.x + s] = APtr[threadIdx.y * K + s + threadIdx.x];
            BShared[threadIdx.y + s][threadIdx.x] = BPtr[(threadIdx.y + s) * N + threadIdx.x];
        }
        __syncthreads();
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += AShared[threadIdx.y][k] * BShared[k][threadIdx.x];
        }
        C[row * N + col] = sum;
    }
}

//使用共享内存 将A的一整行和B的一整列分块存入到sharedMemory中
//每次读一小块进入共享内存，再将这一小块进行矩阵乘法，多次这样操作，最后再把各个小块的结果结合在一起。
template<unsigned int K_>
__global__ void cudaSgemm_V2(float* A,float* B, float* C, int M, int N, int K) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float AShared[K_][K_];
    __shared__ float BShared[K_][K_];
    float* APtr = A + blockIdx.y * blockDim.y * K;
    float* BPtr = B + blockIdx.x * blockDim.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int s = 0; s < K; s += K_) {
            AShared[threadIdx.y][threadIdx.x] = APtr[threadIdx.y * K + s + threadIdx.x];
            BShared[threadIdx.y][threadIdx.x] = BPtr[(threadIdx.y + s) * N + threadIdx.x];
            __syncthreads();
            for (size_t k = 0; k < K_; ++k) {
                sum += AShared[threadIdx.y][k] * BShared[k][threadIdx.x];
            }
            __syncthreads();
        }
        C[row * N + col] = sum;
    }
}

//增加线程的工作量，之前一个线程负责一个点的计算，现在可能一个线程负责多个点的计算
template<unsigned int K_,unsigned int perColNum,unsigned int perRowNum>
__global__ void cudaSgemm_V3(float* A,float* B, float* C, int M, int N, int K) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float AShared[K_][K_];
    __shared__ float BShared[K_][K_];

    for (size_t r = 0; r < perRowNum; ++r) {
        int outRow = row + r * blockDim.y * gridDim.y;
        float* APtr = A + (blockIdx.y + r * gridDim.y) * blockDim.y * K;
        for (size_t c = 0; c < perColNum; ++c) {
            int outCol = col + c * blockDim.x * gridDim.x;
            float* BPtr = B + (blockIdx.x + c * gridDim.x) * blockDim.x;
            float sum = 0.0f;
            for (int s = 0; s < K; s += K_) {
                AShared[threadIdx.y][threadIdx.x] = APtr[threadIdx.y * K + s + threadIdx.x];
                BShared[threadIdx.y][threadIdx.x] = BPtr[(threadIdx.y + s) * N + threadIdx.x];
                __syncthreads();
                for (size_t k = 0; k < K_; ++k) {
                    sum += AShared[threadIdx.y][k] * BShared[k][threadIdx.x];
                }
                __syncthreads();
            }
            C[outRow * N + outCol] = sum;
        }
    }
}

template<unsigned int K_,unsigned int perColNum,unsigned int perRowNum>
__global__ void cudaSgemm_V4(float* A,float* B, float* C, int M, int N, int K) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float AShared[perRowNum][K_][K_];
    __shared__ float BShared[perColNum][K_][K_];

    float registerSum[perRowNum][perColNum];
    for (size_t i = 0; i < perRowNum; ++i)
        for (size_t j = 0; j < perColNum; ++j)
            registerSum[i][j] = 0.0f;

    for (size_t k = 0; k < K; k += K_) {
        for (int r = 0; r < perRowNum; ++r) {
            float* APtr = A + (blockIdx.y + r * gridDim.y) * blockDim.y * K;
            AShared[r][threadIdx.y][threadIdx.x] = APtr[threadIdx.y * K + k + threadIdx.x];
        }
        for (int c = 0; c < perColNum; ++c) {
            float* BPtr = B + (blockIdx.x + c * gridDim.x) * blockDim.x;
            BShared[c][threadIdx.y][threadIdx.x] = BPtr[(threadIdx.y + k) * N + threadIdx.x];
        }
        __syncthreads();
        for (size_t r = 0; r < perRowNum; ++r)
            for (size_t c = 0; c < perColNum; ++c)
                for (size_t s = 0; s < K_; ++s)
                    registerSum[r][c] += AShared[r][threadIdx.y][s] * BShared[c][s][threadIdx.x];
        __syncthreads();
    }

    for (int r = 0; r < perRowNum; ++r)
        for (int c = 0; c < perColNum; ++c) {
            const int outRow = row + r * gridDim.y * blockDim.y;
            const int outCol = col + c * gridDim.x * blockDim.x;
            C[outRow * N + outCol] = registerSum[r][c];
        }
}

template<unsigned int K_,unsigned int perColNum,unsigned int perRowNum>
__global__ void cudaSgemm_V4_1(float* A,float* B, float* C, int M, int N, int K) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float AShared[perRowNum][K_][K_];
    __shared__ float BShared[perColNum][K_][K_];

    float registerSum[perRowNum][perColNum];
    for (size_t i = 0; i < perRowNum; ++i)
        for (size_t j = 0; j < perColNum; ++j)
            registerSum[i][j] = 0.0f;

    for (size_t k = 0; k < K; k += K_) {
        for (size_t r = 0; r < perRowNum; ++r) {
            float* APtr = A + (blockIdx.y + r * gridDim.y) * blockDim.y * K;
            AShared[r][threadIdx.y][threadIdx.x] = APtr[threadIdx.y * K + k + threadIdx.x];
            for (size_t c = 0; c < perColNum; ++c) {
                float* BPtr = B + (blockIdx.x + c * gridDim.x) * blockDim.x;
                BShared[c][threadIdx.y][threadIdx.x] = BPtr[(threadIdx.y + k) * N + threadIdx.x];
                __syncthreads();
                for (size_t s = 0; s < K_; ++s)
                    registerSum[r][c] += AShared[r][threadIdx.y][s] * BShared[c][s][threadIdx.x];
                __syncthreads();
            }
        }
    }

    for (int r = 0; r < perRowNum; ++r)
        for (int c = 0; c < perColNum; ++c) {
            const int outRow = row + r * gridDim.y * blockDim.y;
            const int outCol = col + c * gridDim.x * blockDim.x;
            C[outRow * N + outCol] = registerSum[r][c];
        }
}

//针对V4的性能分析：
// for (r) for (c) for (s) registerSum[r][c] += AShared[r][ty][s] * BShared[c][s][tx];
//每次乘加都需要从共享内存中读取两次，是共享内存带宽瓶颈，且存在反复读取。

// ============ V5_0: 寄存器分块（TM×TN 微块，标量加载，不转置）============
// 在 V4 基础上进一步 tiling：每线程算 TM×TN 微块，循环改成"装寄存器→外积"。
// 本版刻意不用 float4、不转置，聚焦"寄存器复用"这一个优化点。
template<int BM, int BN, int BK, int TM, int TN>
__global__ void cudaSgemm_V5_0(float* A, float* B, float* C, int M, int N, int K) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int threadRow = ty * TM;
    const int threadCol = tx * TN;

    float* APtr = A + by * BM * K;
    float* BPtr = B + bx * BN;
    float* CPtr = C + by * BM * N + bx * BN;

    float acc[TM][TN] = {{0.0f}};

    for (int kb = 0; kb < K; kb += BK) {
        for (int i = ty * blockDim.x + tx; i < BM * BK; i += blockDim.x * blockDim.y) {
            int aRow = i / BK, aCol = i % BK;
            As[aRow][aCol] = APtr[aRow * K + kb + aCol];
        }
        for (int i = ty * blockDim.x + tx; i < BK * BN; i += blockDim.x * blockDim.y) {
            int bRow = i / BN, bCol = i % BN;
            Bs[bRow][bCol] = BPtr[(kb + bRow) * N + bCol];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int m = 0; m < TM; ++m) a[m] = As[threadRow + m][k];
            #pragma unroll
            for (int n = 0; n < TN; ++n) b[n] = Bs[k][threadCol + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a[m] * b[n];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            CPtr[(threadRow + m) * N + threadCol + n] = acc[m][n];
}

// ============ V5: 标准寄存器分块（register tiling + float4 + A 转置）============
template<int BM, int BN, int BK, int TM, int TN>
__global__ void cudaSgemm_V5(float* A, float* B, float* C, int M, int N, int K) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float As[BK][BM];   // A 转置存
    __shared__ float Bs[BK][BN];

    const int threadRow = ty * TM;
    const int threadCol = tx * TN;

    float* APtr = A + by * BM * K;
    float* BPtr = B + bx * BN;
    float* CPtr = C + by * BM * N + bx * BN;

    float acc[TM][TN] = {{0.0f}};

    for (int kb = 0; kb < K; kb += BK) {
        #pragma unroll
        for (int i = ty * blockDim.x + tx; i < BM * BK / 4; i += blockDim.x * blockDim.y) {
            int aRow = i / (BK / 4);
            int aCol = (i % (BK / 4)) * 4;
            float4 v = *reinterpret_cast<const float4*>(&APtr[aRow * K + kb + aCol]);
            As[aCol + 0][aRow] = v.x;  As[aCol + 1][aRow] = v.y;
            As[aCol + 2][aRow] = v.z;  As[aCol + 3][aRow] = v.w;
        }
        #pragma unroll
        for (int i = ty * blockDim.x + tx; i < BK * BN / 4; i += blockDim.x * blockDim.y) {
            int bRow = i / (BN / 4);
            int bCol = (i % (BN / 4)) * 4;
            float4 v = *reinterpret_cast<const float4*>(&BPtr[(kb + bRow) * N + bCol]);
            Bs[bRow][bCol + 0] = v.x;  Bs[bRow][bCol + 1] = v.y;
            Bs[bRow][bCol + 2] = v.z;  Bs[bRow][bCol + 3] = v.w;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float4 a0 = *reinterpret_cast<const float4*>(&As[k][threadRow]);
            float4 a1 = *reinterpret_cast<const float4*>(&As[k][threadRow + 4]);
            float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][threadCol]);
            float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][threadCol + 4]);
            float a[TM] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[TN] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a[m] * b[n];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            CPtr[(threadRow + m) * N + threadCol + n] = acc[m][n];
}

// ============ cp.async 辅助函数（V8 用）============
__device__ __forceinline__ void cpAsync16(void* smem_dst, const void* gmem_src) {
    unsigned int dst = static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem_src));
}
__device__ __forceinline__ void cpAsyncCommit() {
    asm volatile("cp.async.commit_group;\n");
}
template<int N>
__device__ __forceinline__ void cpAsyncWait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ============ V7: 双缓冲（寄存器预取，BK=8 专用）============
template<int BM, int BN, int BK, int TM, int TN>
__global__ void cudaSgemm_V7(float* A, float* B, float* C, int M, int N, int K) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    const int threadRow = ty * TM;
    const int threadCol = tx * TN;

    float* APtr = A + by * BM * K;
    float* BPtr = B + bx * BN;
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

// ============ V8: 双缓冲 + cp.async（动态共享内存，支持任意 BK）============
template<int BM, int BN, int BK, int TM, int TN>
__global__ void cudaSgemm_V8(float* A, float* B, float* C, int M, int N, int K) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    extern __shared__ float4 smem4[];
    float*  As[2];
    float4* Bs[2];
    As[0] = (float*)smem4;               As[1] = As[0] + BM * BK;
    Bs[0] = (float4*)(As[1] + BM * BK);  Bs[1] = Bs[0] + BK * BN / 4;

    const int threadRow = ty * TM;
    const int threadCol = tx * TN;

    float* APtr = A + by * BM * K;
    float* BPtr = B + bx * BN;
    float* CPtr = C + by * BM * N + bx * BN;

    float acc[TM][TN] = {{0.0f}};

    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;

    for (int i = tid; i < BM * BK / 4; i += nthreads) {
        int aRow = i / (BK / 4), aCol = (i % (BK / 4)) * 4;
        cpAsync16(&As[0][aRow * BK + aCol], &APtr[aRow * K + aCol]);
    }
    for (int i = tid; i < BK * BN / 4; i += nthreads) {
        int bRow = i / (BN / 4), bCol = (i % (BN / 4)) * 4;
        cpAsync16((void*)(Bs[0] + bRow * (BN / 4) + bCol / 4), &BPtr[bRow * N + bCol]);
    }
    cpAsyncCommit();
    cpAsyncWait<0>();
    __syncthreads();

    int buf = 0;
    for (int kb = 0; kb < K; kb += BK) {
        if (kb + BK < K) {
            for (int i = tid; i < BM * BK / 4; i += nthreads) {
                int aRow = i / (BK / 4), aCol = (i % (BK / 4)) * 4;
                cpAsync16(&As[1 - buf][aRow * BK + aCol], &APtr[aRow * K + (kb + BK) + aCol]);
            }
            for (int i = tid; i < BK * BN / 4; i += nthreads) {
                int bRow = i / (BN / 4), bCol = (i % (BN / 4)) * 4;
                cpAsync16((void*)(Bs[1 - buf] + bRow * (BN / 4) + bCol / 4), &BPtr[(kb + BK + bRow) * N + bCol]);
            }
            cpAsyncCommit();
        }
        if (kb + BK < K) cpAsyncWait<1>();
        else             cpAsyncWait<0>();
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int m = 0; m < TM; ++m) a[m] = As[buf][(threadRow + m) * BK + k];
            float4 b0 = Bs[buf][k * (BN / 4) + threadCol / 4];
            float4 b1 = Bs[buf][k * (BN / 4) + (threadCol + 4) / 4];
            b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a[m] * b[n];
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

// ============ V9: cp.async + 非转置 A + K 方向 float4 读 ============
template<int BM, int BN, int BK, int TM, int TN>
__global__ void cudaSgemm_V9(float* A, float* B, float* C, int M, int N, int K) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    const int threadRow = ty * TM;
    const int threadCol = tx * TN;

    float* APtr = A + by * BM * K;
    float* BPtr = B + bx * BN;
    float* CPtr = C + by * BM * N + bx * BN;

    float acc[TM][TN] = {{0.0f}};

    const int li = ty * blockDim.x + tx;
    const int aRow = li / (BK / 4);
    const int aCol = (li % (BK / 4)) * 4;
    const int bRow = li / (BN / 4);
    const int bCol = (li % (BN / 4)) * 4;

    cpAsync16(&As[0][aRow][aCol], &APtr[aRow * K + aCol]);
    cpAsync16(&Bs[0][bRow][bCol], &BPtr[bRow * N + bCol]);
    cpAsyncCommit();
    cpAsyncWait<0>();
    __syncthreads();

    int buf = 0;
    for (int kb = 0; kb < K; kb += BK) {
        if (kb + BK < K) {
            cpAsync16(&As[1 - buf][aRow][aCol], &APtr[aRow * K + (kb + BK) + aCol]);
            cpAsync16(&Bs[1 - buf][bRow][bCol], &BPtr[(kb + BK + bRow) * N + bCol]);
            cpAsyncCommit();
        }
        if (kb + BK < K) cpAsyncWait<1>();
        else             cpAsyncWait<0>();
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k += 4) {
            float a[TM][4], b[4][TN];
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                float4 av = *reinterpret_cast<const float4*>(&As[buf][threadRow + m][k]);
                a[m][0] = av.x; a[m][1] = av.y; a[m][2] = av.z; a[m][3] = av.w;
            }
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][k + kk][threadCol]);
                float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][k + kk][threadCol + 4]);
                b[kk][0] = b0.x; b[kk][1] = b0.y; b[kk][2] = b0.z; b[kk][3] = b0.w;
                b[kk][4] = b1.x; b[kk][5] = b1.y; b[kk][6] = b1.z; b[kk][7] = b1.w;
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    #pragma unroll
                    for (int kk = 0; kk < 4; ++kk)
                        acc[m][n] += a[m][kk] * b[kk][n];
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

// ============ WMMA Tensor Core 常量 ============
constexpr int WarpSize = 32;
constexpr int TMMA_TM = 16;
constexpr int TMMA_TN = 16;
constexpr int TMMA_TK = 8;
constexpr int WARP_M = 4, WARP_N = 2;
constexpr int TBM = WARP_M * TMMA_TM;
constexpr int TBN = WARP_N * TMMA_TN;
constexpr int TBK = 32;

// ============ T0: 最朴素的 WMMA TF32（直接读全局内存）============
__global__ void cudaSgemm_T0(float* A, float* B, float* C, int M, int N, int K) {
    const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) / WarpSize;
    const int n_tiles = N / TMMA_TN;
    const int c_row = (gwarp / n_tiles) * TMMA_TM;
    const int c_col = (gwarp % n_tiles) * TMMA_TN;

    wmma::fragment<wmma::matrix_a, TMMA_TM, TMMA_TN, TMMA_TK, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TMMA_TM, TMMA_TN, TMMA_TK, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TMMA_TM, TMMA_TN, TMMA_TK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += TMMA_TK) {
        wmma::load_matrix_sync(a_frag, A + c_row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + c_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
}

//使用共享内存分块 对应 V4：block 分块 + warp 从 smem 加载 fragment
__global__ void cudaSgemm_T1(float* A, float* B, float* C, int M, int N, int K) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WarpSize;      // block 内 warp 编号 0..7
    const int warp_m  = warp_id / WARP_N;    // 0..3
    const int warp_n  = warp_id % WARP_N;    // 0..1

    const int bx = blockIdx.x, by = blockIdx.y;

    __shared__ float As[TBM][TBK];   // 64×32 = 8KB
    __shared__ float Bs[TBK][TBN];   // 32×32 = 4KB

    float* APtr = A + by * TBM * K;
    float* BPtr = B + bx * TBN;

    wmma::fragment<wmma::matrix_a, TMMA_TM, TMMA_TN, TMMA_TK, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TMMA_TM, TMMA_TN, TMMA_TK, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TMMA_TM, TMMA_TN, TMMA_TK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int kb = 0; kb < K; kb += TBK) {
        // float4 协作加载 A tile（64×32 = 512 个 float4，每线程 2 个）
        for (int i = tid; i < TBM * TBK / 4; i += blockDim.x) {
            int r = i / (TBK / 4);
            int c = (i % (TBK / 4)) * 4;
            float4 v = *reinterpret_cast<const float4*>(&APtr[r * K + kb + c]);
            As[r][c + 0] = v.x;  As[r][c + 1] = v.y;
            As[r][c + 2] = v.z;  As[r][c + 3] = v.w;
        }
        // float4 协作加载 B tile（32×32 = 256 个 float4，每线程 1 个）
        for (int i = tid; i < TBK * TBN / 4; i += blockDim.x) {
            int r = i / (TBN / 4);
            int c = (i % (TBN / 4)) * 4;
            float4 v = *reinterpret_cast<const float4*>(&BPtr[(kb + r) * N + c]);
            Bs[r][c + 0] = v.x;  Bs[r][c + 1] = v.y;
            Bs[r][c + 2] = v.z;  Bs[r][c + 3] = v.w;
        }
        __syncthreads();

        // 每个 warp 沿 K 循环 mma
        for (int k = 0; k < TBK; k += TMMA_TK) {
            wmma::load_matrix_sync(a_frag, &As[warp_m * TMMA_TM][k], TBK);
            wmma::load_matrix_sync(b_frag, &Bs[k][warp_n * TMMA_TN], TBN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    const int c_row = (by * WARP_M + warp_m) * TMMA_TM;
    const int c_col = (bx * WARP_N + warp_n) * TMMA_TN;
    wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
}



int main() {
    printf("SGEMM %dx%dx%d, block = %dx%d\n\n", M, N, K, BLOCK, BLOCK);

    CudaStepTimer gpu_timer;
    HostStepTimer host_timer;

    // 1. 初始化 CUDA 设备
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (compute capability %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // 2. 分配主机内存
    host_timer.begin();
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C_cpu = (float*)malloc(M * N * sizeof(float));
    float* h_C_gpu = (float*)malloc(M * N * sizeof(float));
    host_timer.end("Host malloc (A, B, C_cpu, C_gpu)");

    // 3. 初始化矩阵 A 和 B
    host_timer.begin();
    randomMatrix(h_A, M, K);
    randomMatrix(h_B, K, N);
    memset((void*)h_C_cpu, 0, M * N * sizeof(float));
    memset((void*)h_C_gpu, 0, M * N * sizeof(float));
    host_timer.end("Host random init");

    // 4. 分配设备内存
    gpu_timer.begin();
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    gpu_timer.end("cudaMalloc (A, B, C)");

    // 5. 将矩阵 A 和 B 从主机复制到设备
    gpu_timer.begin();
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    gpu_timer.end("cudaMemcpy H2D (A, B)");

    // 用 cuBLAS 算参考结果（大矩阵下 CPU 朴素版太慢）
    cublasHandle_t cublas_handle = nullptr;
    cublasCreate(&cublas_handle);
    printf("\n");
    gpu_timer.begin();
    cublasSgemmRowMajor(cublas_handle, d_A, d_B, d_C);   // d_C = A * B
    cudaMemcpy(h_C_cpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    const double cpu_ms = gpu_timer.end("cuBLAS reference");
    //V0
    {
    // 6. 在设备上执行矩阵乘法
    // grid.x 覆盖输出列（N 方向，对应 threadIdx.x），grid.y 覆盖输出行（M 方向）
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    gpu_timer.begin();
    cudaSgemm_V0<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V0 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    // 7. 将结果从设备复制回主机
    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    // 9. 结果比对（以 CPU 结果为基准）
    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V0 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //V1-globalMemory
    // {
    // // 6. 在设备上执行矩阵乘法
    // // grid.x 覆盖输出列（N 方向，对应 threadIdx.x），grid.y 覆盖输出行（M 方向）
    // dim3 block(BLOCK, BLOCK);
    // dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    // gpu_timer.begin();
    // cudaSgemm_V1<BLOCK,K><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    // const float kernel_ms = gpu_timer.end("cudaSgemm_V0 kernel");

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error: %s\n", cudaGetErrorString(err));
    //     return -1;
    // }
    // if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
    //     printf("CUDA sync error: %s\n", cudaGetErrorString(err));
    //     return -1;
    // }
    // printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
    //        2.0 * M * N * K / (kernel_ms * 1e6));

    // // 7. 将结果从设备复制回主机
    // gpu_timer.begin();
    // cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // gpu_timer.end("cudaMemcpy D2H (C)");

    // // 9. 结果比对（以 CPU 结果为基准）
    // printf("\n");
    // host_timer.begin();
    // compareMatrix(h_C_cpu, h_C_gpu, M, N);
    // host_timer.end("compareMatrix");

    // printf("\n[cudaSgemm_V1 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
    //        cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
    //        kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
    //        cpu_ms / kernel_ms);

    // }

    //V2
    {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    gpu_timer.begin();
    cudaSgemm_V2<BLOCK><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V2 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    // 7. 将结果从设备复制回主机
    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    // 9. 结果比对（以 CPU 结果为基准）
    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V2 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //V3
    {
    // 6. 在设备上执行矩阵乘法
    // grid.x 覆盖输出列（N 方向，对应 threadIdx.x），grid.y 覆盖输出行（M 方向）
    const int perRowNum = 2;
    const int perColNum = 2;

    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK / perColNum, (M + BLOCK - 1) / BLOCK / perRowNum);

    gpu_timer.begin();
    cudaSgemm_V3<BLOCK,perColNum,perRowNum><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V3 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    // 7. 将结果从设备复制回主机
    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    // 9. 结果比对（以 CPU 结果为基准）
    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V3 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }
    
     //V4
    {
    // 6. 在设备上执行矩阵乘法
    // grid.x 覆盖输出列（N 方向，对应 threadIdx.x），grid.y 覆盖输出行（M 方向）
    const int perRowNum = 4;
    const int perColNum = 4;

    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK / perColNum, (M + BLOCK - 1) / BLOCK / perRowNum);

    gpu_timer.begin();
    cudaSgemm_V4<BLOCK,perColNum,perRowNum><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V4 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    // 7. 将结果从设备复制回主机
    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    // 9. 结果比对（以 CPU 结果为基准）
    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V4 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

     {
    // 6. 在设备上执行矩阵乘法
    // grid.x 覆盖输出列（N 方向，对应 threadIdx.x），grid.y 覆盖输出行（M 方向）
    const int perRowNum = 2;
    const int perColNum = 2;

    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK / perColNum, (M + BLOCK - 1) / BLOCK / perRowNum);

    gpu_timer.begin();
    cudaSgemm_V4_1<BLOCK,perColNum,perRowNum><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V4_1 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    // 7. 将结果从设备复制回主机
    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    // 9. 结果比对（以 CPU 结果为基准）
    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V4_1 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //V5_0：寄存器分块（TM×TN 微块，A 不转置）
    // {
    // // BM BN 表示一个线程 block 要负责的输出 tile
    // // TM TN 表示每个线程要算的微块大小
    // const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;

    // dim3 block(BN / TN, BM / TM);   // 16 x 16 = 256 线程
    // dim3 grid(N / BN, M / BM);      // 32 x 32

    // gpu_timer.begin();
    // cudaSgemm_V5_0<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    // const float kernel_ms = gpu_timer.end("cudaSgemm_V5_0 kernel");

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error: %s\n", cudaGetErrorString(err));
    //     return -1;
    // }
    // if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
    //     printf("CUDA sync error: %s\n", cudaGetErrorString(err));
    //     return -1;
    // }
    // printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
    //        2.0 * M * N * K / (kernel_ms * 1e6));

    // // 将结果从设备复制回主机
    // gpu_timer.begin();
    // cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // gpu_timer.end("cudaMemcpy D2H (C)");

    // // 结果比对（以 cuBLAS 结果为基准）
    // printf("\n");
    // host_timer.begin();
    // compareMatrix(h_C_cpu, h_C_gpu, M, N);
    // host_timer.end("compareMatrix");

    // printf("\n[cudaSgemm_V5_0 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
    //        cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
    //        kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
    //        cpu_ms / kernel_ms);
    // }

    //V5
    {
    //BM BN 表示一个线程block要负责的元素个数
    //BK 表示K维度的分块大小
    const int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8;

    dim3 block(BN / TM, BM / TN);   // 16 x 16 = 256 线程
    dim3 grid(N / BN, M / BM);    // 32 x 32

    gpu_timer.begin();
    cudaSgemm_V5<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V5 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    // 7. 将结果从设备复制回主机
    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    // 9. 结果比对（以 CPU 结果为基准）
    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V5 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //V5 BK=16 调参
    {
    const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;

    dim3 block(BN / TN, BM / TM);   // 16 x 16 = 256 线程
    dim3 grid(N / BN, M / BM);      // 32 x 32

    gpu_timer.begin();
    cudaSgemm_V5<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V5 BK=16 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V5 BK=16 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //V5 BK=32 调参
    {
    const int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8;

    dim3 block(BN / TN, BM / TM);   // 16 x 16 = 256 线程
    dim3 grid(N / BN, M / BM);      // 32 x 32

    gpu_timer.begin();
    cudaSgemm_V5<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V5 BK=32 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V5 BK=32 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //V7：双缓冲（double buffering，BK=8）
    {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    dim3 block(BN / TN, BM / TM);   // 16 x 16 = 256 线程
    dim3 grid(N / BN, M / BM);      // 32 x 32

    gpu_timer.begin();
    cudaSgemm_V7<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V7 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V7 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //V8：双缓冲 + cp.async
    {
    const int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8;
    const int smem_size = (2 * BM * BK + 2 * BK * BN) * sizeof(float);   // 64KB
    cudaFuncSetAttribute(cudaSgemm_V8<BM, BN, BK, TM, TN>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    dim3 block(BN / TN, BM / TM);   // 16 x 16 = 256 线程
    dim3 grid(N / BN, M / BM);      // 32 x 32

    gpu_timer.begin();
    cudaSgemm_V8<BM, BN, BK, TM, TN><<<grid, block, smem_size>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V8 BK=32 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V8 BK=32 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //V9：cp.async + 非转置 A + K 方向 float4 读
    {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    dim3 block(BN / TN, BM / TM);   // 16 x 16 = 256 线程
    dim3 grid(N / BN, M / BM);      // 32 x 32

    gpu_timer.begin();
    cudaSgemm_V9<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_V9 kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N);
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_V9 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //T0：最朴素的 WMMA TF32（无共享内存）
    {
    dim3 block(256);                          // 8 warps
    dim3 grid((M / 16) * (N / 16) / 8);       // 65536 个 tile / 8 = 8192 块

    gpu_timer.begin();
    cudaSgemm_T0<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_T0 (TF32 naive) kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N, 5e-3);   // TF32 精度 ~1e-3
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_T0 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    //T1：WMMA TF32 Tensor Core（共享内存分块）
    {
    dim3 block(256);                     // 8 warps
    dim3 grid(N / TBN, M / TBM);         // 128 x 64 = 8192 块

    gpu_timer.begin();
    cudaSgemm_T1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    const float kernel_ms = gpu_timer.end("cudaSgemm_T1 (TF32) kernel");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("  [GPU] %-32s %9.2f GFLOP/s\n", "kernel throughput",
           2.0 * M * N * K / (kernel_ms * 1e6));

    gpu_timer.begin();
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_timer.end("cudaMemcpy D2H (C)");

    printf("\n");
    host_timer.begin();
    compareMatrix(h_C_cpu, h_C_gpu, M, N, 5e-3);   // TF32 精度 ~1e-3，用宽松容差
    host_timer.end("compareMatrix");

    printf("\n[cudaSgemm_T1 Summary] CPU: %.3f ms (%.2f GFLOP/s), GPU kernel: %.3f ms (%.2f GFLOP/s), speedup: %.2fx\n",
           cpu_ms, 2.0 * M * N * K / (cpu_ms * 1e6),
           kernel_ms, 2.0 * M * N * K / (kernel_ms * 1e6),
           cpu_ms / kernel_ms);
    }

    // 10. 释放内存
    gpu_timer.begin();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    gpu_timer.end("cudaFree");

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    cublasDestroy(cublas_handle);
    return 0;
}
