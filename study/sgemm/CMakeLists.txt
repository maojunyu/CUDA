# SGEMM 学习基准：kernel.cu 内含 main，依次计时 V0~V9、T0/T1 并比对。
# 最优版本（当前 V7）已晋升到 ops/src/sgemm.cu。
add_executable(sgemm_bench kernel.cu)

target_compile_features(sgemm_bench PRIVATE cxx_std_17 cuda_std_17)

if(MSVC)
    target_compile_options(sgemm_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

set_target_properties(sgemm_bench PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_link_libraries(sgemm_bench PRIVATE CUDA::cudart CUDA::cublas)

# Windows 下 cuBLAS 为动态链接，构建后自动拷贝 DLL 到可执行文件旁
if(WIN32)
    file(GLOB CUBLAS_RUNTIME_DLLS
        "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
        "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll"
    )
    if(CUBLAS_RUNTIME_DLLS)
        add_custom_command(TARGET sgemm_bench POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CUBLAS_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:sgemm_bench>"
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
    endif()
endif()
