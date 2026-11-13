# 工具链冒烟测试：SAXPY（y = a*x + y），验证 CUDA 工具链配置成功。
add_executable(toolchain_saxpy kernel.cu)

target_compile_features(toolchain_saxpy PRIVATE cxx_std_17 cuda_std_17)

if(MSVC)
    target_compile_options(toolchain_saxpy PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

set_target_properties(toolchain_saxpy PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_link_libraries(toolchain_saxpy PRIVATE CUDA::cudart)
