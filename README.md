# CUDA 算子优化项目

项目分**两块**：

| 目录 | 作用 | 产物 |
|------|------|------|
| `ops/` | **算子优化库**（用于大模型推理优化的 DLL） | `cuda_ops.dll` |
| `study/` | **算子学习区**（每个算子一个目录，保留所有实验版本 + 基准测试） | 各基准 `.exe` |

工作流：**在 `study/<算子>/` 里学习、调参、计时 → 跑出最优版本 → 把它"晋升"到 `ops/src/<算子>.cu`**，对外 C API 保持不变。

## 你的环境

| 项目 | 值 |
|------|----|
| GPU | NVIDIA GeForce RTX 5070 Ti（Blackwell，计算能力 **sm_120**） |
| CUDA Toolkit | 13.5（`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5`） |
| 编译工具 | Visual Studio 2026 Community（MSVC 19.51） |
| CMake | 4.5（`E:\Tools\CMake\bin`） |

## 目录结构

```
E:/AI_Infra/CUDA
├── CMakeLists.txt              # 顶层：声明语言、find_package(CUDAToolkit)、默认架构
├── CMakePresets.json           # 预设：VS 2026 + x64 + sm_120
│
├── ops/                        # ① 算子优化库（DLL）
│   ├── CMakeLists.txt          # add_library(cuda_ops SHARED)
│   ├── include/cuda_ops/       # 对外 C API（稳定的 DLL 接口）
│   │   ├── cuda_ops.h          #   导出宏 + 版本号
│   │   ├── sgemm.h             #   SGEMM
│   │   └── flash_attention.h   #   FlashAttention
│   ├── src/                    # 各算子的"最优版本"实现
│   │   ├── sgemm.cu            #   SGEMM V7（双缓冲 + 转置 A + float4）
│   │   └── flash_attention.cu  #   FlashAttention 前向（在线 softmax）
│   └── tests/                  # DLL 冒烟测试（test_ops.exe）
│
└── study/                      # ② 算子学习区
    ├── 00_toolchain/           # 工具链冒烟测试（SAXPY）
    ├── sgemm/                  # SGEMM 学习：V0~V9 + Tensor Core T0/T1 + 笔记
    ├── cublas/                 # cuBLAS 参考基准
    └── flash_attention/        # FlashAttention 学习
```

## 在 VSCode 中使用

1. 安装扩展：**CMake Tools**、**C/C++**、推荐 **Nsight VSCode Edition**（`.cu` 语法高亮与调试）。
2. 用 VSCode 打开 `E:/AI_Infra/CUDA`（顶层目录）。
3. 底部状态栏选择：
   - 工具包（Kit）：**Visual Studio Community 2026**（MSVC x64）
   - 配置预设（Configure Preset）：**windows-msvc-cuda**
   - 构建类型（Build Variant）：`Release`
4. 按 `F7`（或 `CMake: Build`）构建。

## 产物与运行

```
build/windows-msvc-cuda/lib/cuda_ops.dll                          # 算子库
build/windows-msvc-cuda/ops/tests/Release/test_ops.exe            # DLL 冒烟测试
build/windows-msvc-cuda/study/sgemm/Release/sgemm_bench.exe       # SGEMM 学习基准
build/windows-msvc-cuda/study/flash_attention/Release/flash_attention_bench.exe
```

- 冒烟测试会打印每个算子的 `PASS/FAIL`，验证 DLL 对外 API 正确。
- 学习基准打印各版本的耗时与 GFLOP/s，用于版本对比。

## 新增一个算子（标准流程）

1. **学习**：在 `study/` 下建 `study/<算子名>/`，写 `kernel.cu`（含 `main`）与 `CMakeLists.txt`，
   在顶层 `study/CMakeLists.txt` 里 `add_subdirectory`。
2. **调优**：在该目录里逐版本迭代，用 benchmark 对比，写优化笔记（参考 `study/sgemm/*.md`）。
3. **晋升**：把最优 kernel 拷到 `ops/src/<算子名>.cu`，在 `ops/include/cuda_ops/<算子名>.h`
   里声明 C API，在 `ops/CMakeLists.txt` 里加入该源文件。
4. **验证**：在 `ops/tests/test_ops.cu` 里加对应校验，构建后跑 `test_ops.exe`。

## 命令行手动构建

```bat
cmake -S E:/AI_Infra/CUDA -B build -G "Visual Studio 18 2026" -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release
build\windows-msvc-cuda\ops\tests\Release\test_ops.exe
```

## 常用修改

### 更换目标 GPU 架构
`CMakePresets.json` 里改 `"CMAKE_CUDA_ARCHITECTURES": "120"`，例如：
- 本机 RTX 5070 Ti：`"120"`
- 兼容多代 GPU：`"75;80;86;89;90;100;120"`
- 自动检测：`"native"`

### 关于中文注释
MSVC 默认按 GBK（代码页 936）读取源码，中文注释会报 `C4819` 警告甚至错行。
各 `CMakeLists.txt` 已统一加 `/utf-8` 编译选项，源文件请保存为 **UTF-8** 编码。

### DLL 部署注意
`cuda_ops.dll` 静态链接了 CUDA 运行时（不依赖 `cudart64_*.dll`），
但 **cuBLAS 仍是动态链接**，部署到推理运行时需一并带上 `cublas64_*.dll`。
