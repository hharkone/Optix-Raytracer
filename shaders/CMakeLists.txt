# Compile devicePrograms.cu to PTX for OptiX JIT compilation at runtime.
# CUDA_PTX_COMPILATION ON  → outputs .ptx instead of .obj
# 75-virtual              → PTX only, no embedded cubin (required for OptiX)
# CUDA_SEPARABLE_COMPILATION OFF → PTX programs are self-contained entry points

add_library(device_programs OBJECT
    devicePrograms.cu
)

set_property(TARGET device_programs PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET device_programs PROPERTY CUDA_ARCHITECTURES "75-virtual")
set_property(TARGET device_programs PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

target_include_directories(device_programs PRIVATE
    ${CMAKE_SOURCE_DIR}/shaders
    ${OptiX_INCLUDE_DIR}
)
