# ① 算子优化库：编译为共享库（Windows 下为 cuda_ops.dll）。
#
# 这里只放各算子的"最优版本"。学习区（study/）跑出更优版本后，
# 把 kernel 实现拷到 src/ 对应文件、更新头文件注释即可，对外 API 保持不变。

add_library(cuda_ops SHARED
    src/sgemm.cu
    src/flash_attention.cu
)

# 构建库时导出符号（dllimport/dllexport 见 include/cuda_ops/cuda_ops.h）
target_compile_definitions(cuda_ops PRIVATE CUDA_OPS_BUILD)

target_compile_features(cuda_ops PUBLIC cxx_std_17 cuda_std_17)

# 公开头文件目录（使用者只需 include <cuda_ops/xxx.h>）
target_include_directories(cuda_ops PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 静态链接 CUDA 运行时，使 DLL 不依赖 cudart64_*.dll，便于部署到推理运行时。
# 注意：cuBLAS 仍是动态链接（cublas64_*.dll），部署时需一并带上。
set_target_properties(cuda_ops PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Static
)

# MSVC 默认按 GBK(代码页936) 读取源文件，强制 UTF-8 解析
if(MSVC)
    target_compile_options(cuda_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

target_link_libraries(cuda_ops PUBLIC CUDA::cudart CUDA::cublas)

# DLL / 导入库统一输出到 build/<preset>/lib，便于测试与部署
set_target_properties(cuda_ops PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# 库的冒烟测试（验证 DLL 对外 API 可用）
add_subdirectory(tests)
