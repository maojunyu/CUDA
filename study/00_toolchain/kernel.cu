// 最小可运行示例：SAXPY（y = a*x + y），用于验证 CUDA 工具链是否配置成功。
// 你后续可以把这里的代码替换成自己的 SGEMM 实现。

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void saxpy_kernel(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    // 打印设备信息
    int device = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);

    const int n = 1 << 20;               // 1M 个元素
    const size_t bytes = n * sizeof(float);

    float* hx = static_cast<float*>(malloc(bytes));
    float* hy = static_cast<float*>(malloc(bytes));
    for (int i = 0; i < n; ++i) {
        hx[i] = 1.0f;
        hy[i] = 2.0f;
    }

    float *dx = nullptr, *dy = nullptr;
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    saxpy_kernel<<<blocks, threads>>>(n, 3.0f, dx, dy);

    cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost);

    // 校验：y = 3*1 + 2 = 5
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = fmaxf(max_err, fabsf(hy[i] - 5.0f));
    }
    printf("SAXPY max error: %f\n", max_err);
    if (max_err < 1e-4f) {
        printf("CUDA toolchain OK!\n");
    }

    cudaFree(dx);
    cudaFree(dy);
    free(hx);
    free(hy);
    return 0;
}
