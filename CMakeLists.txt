cmake_minimum_required(VERSION 3.24)

# CUDA 作为语言参与编译；CXX 用于宿主代码
project(cuda_ops_project LANGUAGES CXX CUDA)

# 定位 NVIDIA CUDA Toolkit（nvcc、头文件、库）
find_package(CUDAToolkit REQUIRED)

# 目标 GPU 架构：RTX 5070 Ti = Blackwell = sm_120
# 也可以在 CMakePresets.json 或命令行里用 -DCMAKE_CUDA_ARCHITECTURES=... 覆盖
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

message(STATUS "CUDA Toolkit  : ${CUDAToolkit_VERSION}")
message(STATUS "nvcc          : ${CUDAToolkit_NVCC_EXECUTABLE}")
message(STATUS "CUDA arch     : ${CMAKE_CUDA_ARCHITECTURES}")

# 项目分两块：
#   ① ops/   算子优化库（编译为 DLL，对外发布，只放各算子最优版本）
#   ② study/ 算子学习区（每个算子一个目录，保留所有实验版本 + 基准测试）
add_subdirectory(ops)
add_subdirectory(study)
