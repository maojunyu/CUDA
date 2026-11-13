# ② 算子学习区：每个算子一个目录，编译为独立的基准可执行文件（含 main）。
#
# 这里保留算子的所有实验版本（如 SGEMM 的 V0~V9、T0/T1），
# 用于调参、计时、与 CPU/cuBLAS 参考比对。跑出最优版本后，
# 把它"晋升"到 ops/src/ 对应文件，对外 API 保持不变。

add_subdirectory(00_toolchain)      # 工具链冒烟测试（SAXPY）
add_subdirectory(sgemm)             # SGEMM 学习：V0~V9 + Tensor Core T0/T1
add_subdirectory(cublas)            # cuBLAS 参考基准
add_subdirectory(flash_attention)   # FlashAttention 学习
