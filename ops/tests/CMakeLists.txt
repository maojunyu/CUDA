# cuda_ops 库的冒烟测试：链接 DLL，用 CPU 参考实现校验各算子正确性。

add_executable(test_ops test_ops.cu)
target_compile_features(test_ops PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(test_ops PRIVATE cuda_ops)

set_target_properties(test_ops PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

if(MSVC)
    target_compile_options(test_ops PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

# 运行时依赖：把 cuda_ops.dll 与 cublas64_*.dll 拷到测试可执行文件旁
if(WIN32)
    add_custom_command(TARGET test_ops POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:cuda_ops>"
            "$<TARGET_FILE_DIR:test_ops>"
        VERBATIM)

    file(GLOB CUBLAS_RUNTIME_DLLS
        "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
        "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll"
    )
    if(CUBLAS_RUNTIME_DLLS)
        add_custom_command(TARGET test_ops POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CUBLAS_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:test_ops>"
            COMMAND_EXPAND_LISTS
            VERBATIM
        )
    endif()
endif()
