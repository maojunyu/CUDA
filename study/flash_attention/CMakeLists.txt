# FlashAttention 学习基准：kernel.cu 内含 main，前向 + CPU 参考比对 + 计时。
# 当前版本已晋升到 ops/src/flash_attention.cu。
add_executable(flash_attention_bench kernel.cu)

target_compile_features(flash_attention_bench PRIVATE cxx_std_17 cuda_std_17)

if(MSVC)
    target_compile_options(flash_attention_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

set_target_properties(flash_attention_bench PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_link_libraries(flash_attention_bench PRIVATE CUDA::cudart)
