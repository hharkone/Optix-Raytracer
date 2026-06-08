# Compile device_programs.cu to PTX for OptiX JIT compilation at runtime.
# CUDA_PTX_COMPILATION ON  → outputs .ptx instead of .obj
# 75-virtual              → PTX only, no embedded cubin (required for OptiX)
# CUDA_SEPARABLE_COMPILATION OFF → PTX programs are self-contained entry points

add_library(device_programs OBJECT
    device_programs.cu
    device_math.h
    launch_params.h
    scene_data.h
)

source_group("Header Files" FILES device_math.h launch_params.h scene_data.h)
source_group("Source Files" FILES device_programs.cu)

set_property(TARGET device_programs PROPERTY CUDA_PTX_COMPILATION ON)
set_property(TARGET device_programs PROPERTY CUDA_ARCHITECTURES "75-virtual")
set_property(TARGET device_programs PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

target_include_directories(device_programs PRIVATE
    ${CMAKE_SOURCE_DIR}/shaders
    ${OptiX_INCLUDE_DIR}
)

# VS IntelliSense reads IncludePath, not CudaCompile/Include, so inject a props
# file that adds the OptiX directory to IncludePath for all configurations.
if(MSVC)
    configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/cuda_intellisense.props.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cuda_intellisense.props"
        @ONLY
    )
    set_target_properties(device_programs PROPERTIES
        VS_USER_PROPS "${CMAKE_CURRENT_BINARY_DIR}/cuda_intellisense.props"
    )
endif()
