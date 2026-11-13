# cuBLAS 参考基准：gemm.cu 内含 main，跑 cublasSgemm(FP32) 与 cublasGemmEx(TF32)。
add_executable(cublas_gemm gemm.cu)

target_compile_features(cublas_gemm PRIVATE cxx_std_17 cuda_std_17)

if(MSVC)
    target_compile_options(cublas_gemm PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

set_target_properties(cublas_gemm PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_link_libraries(cublas_gemm PRIVATE CUDA::cudart CUDA::cublas)

# Windows 下 cuBLAS 为动态链接，构建后自动拷贝 DLL 到可执行文件旁
if(WIN32)
    file(GLOB CUBLAS_RUNTIME_DLLS
        "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
        "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll"
    )
    if(CUBLAS_RUNTIME_DLLS)
        add_custom_command(TARGET cublas_gemm POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CUBLAS_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:cublas_gemm>"
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
    endif()
endif()
