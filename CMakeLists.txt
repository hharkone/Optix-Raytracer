cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# ── Policy guards ────────────────────────────────────────────────────────────
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW) # FetchContent timestamp behaviour
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW) # find_package honours <Pkg>_ROOT variables
endif()

# ── CUDA architectures (must be set before project()) ────────────────────────
# sm_75 = Turing (RTX 20xx)  sm_86 = Ampere (RTX 30xx)  sm_89 = Ada (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# ── Project ──────────────────────────────────────────────────────────────────
project(OptixRaytracer
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Place all runtime outputs under build/bin/<Config>/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" CFG)
    set("CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG}" "${CMAKE_BINARY_DIR}/bin/${cfg}")
endforeach()

# ── Custom CMake modules ──────────────────────────────────────────────────────
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ── System / SDK dependencies ─────────────────────────────────────────────────
find_package(OptiX     REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL    REQUIRED)

# ── Third-party via FetchContent ──────────────────────────────────────────────
include(FetchContent)

# GLFW 3.4 — windowing
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ImGui docking branch — UI with docking + multi-viewport support
# GIT_SHALLOW FALSE is required when targeting a branch rather than a tag
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        docking
    GIT_SHALLOW    FALSE
)
FetchContent_MakeAvailable(imgui)

# ImGui has no CMakeLists.txt — create the STATIC target manually
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
target_compile_options(imgui PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W0>)

# GLAD — pre-generated OpenGL 3.3 core function loader (committed to repo)
add_subdirectory(extern/glad)

# ── Project subdirectories ────────────────────────────────────────────────────
add_subdirectory(shaders)   # device_programs OBJECT library (PTX)
add_subdirectory(src)       # OptixRaytracer executable
