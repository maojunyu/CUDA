#pragma once

// cuda_ops 统一入口头文件。
// 构建库时（见 ops/CMakeLists.txt）会定义 CUDA_OPS_BUILD，从而导出符号；
// 使用库的一方不定义它，自动走 __declspec(dllimport)。
#ifdef _WIN32
  #ifdef CUDA_OPS_BUILD
    #define CUDA_OPS_API __declspec(dllexport)
  #else
    #define CUDA_OPS_API __declspec(dllimport)
  #endif
#else
  #define CUDA_OPS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 库版本字符串，例如 "0.1.0"。
CUDA_OPS_API const char* cuda_ops_version(void);

#ifdef __cplusplus
}
#endif
