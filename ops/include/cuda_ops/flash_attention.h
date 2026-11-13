#pragma once

#include "cuda_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// FlashAttention 前向（float32）。
//
// Q/K/V/O 布局：[batch × heads, seq_len, head_dim]，行主序。
// 当前最优版本只支持 head_dim == 64（其余值返回 cudaErrorInvalidValue）。
// softmax_scale 通常传 1/sqrt(head_dim)。
// causal = 1 表示因果掩码（自回归解码），0 表示全注意力。
// stream 传 cudaStream_t（强转成 void*），0 或 nullptr 表示默认流。
//
// 返回 0 表示成功，否则为 cudaError_t 错误码。
CUDA_OPS_API int cuda_ops_flash_attention_fwd(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int batch, int heads, int seq_len, int head_dim,
    float softmax_scale, int causal,
    void* stream);

#ifdef __cplusplus
}
#endif
