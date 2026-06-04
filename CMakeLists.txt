cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# ── Policy guards ────────────────────────────────────────────────────────────
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # FetchContent timestamp behaviour
endif()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW) # find_package honours <Pkg>_ROOT variables
endif()
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD) # Allow FetchContent_Populate (deprecated in CMake 4.x)
endif()

# Third-party dependencies declare cmake_minimum_required versions < 3.10, which
# triggers a CMake 4.x deprecation warning that we cannot fix in their source.
# Suppress it globally — our own CMake code does not use deprecated features.
set(CMAKE_WARN_DEPRECATED OFF CACHE INTERNAL "")

# ── CUDA architectures (must be set before project()) ────────────────────────
# sm_75 = Turing (RTX 20xx)  sm_86 = Ampere (RTX 30xx)  sm_89 = Ada (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# ── Project ──────────────────────────────────────────────────────────────────
project(OptixRaytracer
    VERSION 0.1.0
    LANGUAGES CXX CUDA C  # C is needed for miniz (tinyexr); declared here so detection
)                         # runs upfront rather than being triggered by GLFW's nested project()

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
find_package(Vulkan    REQUIRED)

# ── Third-party via FetchContent ──────────────────────────────────────────────
include(FetchContent)

# On Windows/MSVC, pthreads do not exist.  Pre-populate the three cache
# variables that CMake's FindThreads module creates when its compile/link
# probes fail.  The check_*() macros skip any test whose result variable is
# already defined in the cache, so this avoids three slow probe round-trips
# (compile + link) on every clean configure.  FindThreads still finds
# Threads::Threads via the unconditional Windows fallback at the end of the
# module.  Must be set before any FetchContent_MakeAvailable call that
# internally calls find_package(Threads) — e.g. GLFW.
if(WIN32)
    set(CMAKE_HAVE_LIBC_PTHREAD    0 CACHE INTERNAL "pthread in libc (pre-populated: N/A on MSVC)")
    set(CMAKE_HAVE_PTHREADS_CREATE 0 CACHE INTERNAL "pthreads lib    (pre-populated: N/A on MSVC)")
    set(CMAKE_HAVE_PTHREAD_CREATE  0 CACHE INTERNAL "pthread lib     (pre-populated: N/A on MSVC)")
endif()

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
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)
target_compile_options(imgui PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W0>)

# ImGuizmo — 3D gizmo (translate / rotate / scale) overlay for Dear ImGui.
# No CMakeLists.txt; compiled as part of the imgui STATIC target so it
# shares ImGui's context and link dependencies automatically.
FetchContent_Declare(imguizmo
    GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(imguizmo)
if(NOT imguizmo_POPULATED)
    FetchContent_Populate(imguizmo)
endif()
target_sources(imgui PRIVATE ${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp)
target_include_directories(imgui PUBLIC ${imguizmo_SOURCE_DIR}/src)

# tinygltf v2.8.21 — glTF 2.0 loader (header-only; implementation defined in SceneLoader.cpp)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.8.21
    GIT_SHALLOW    TRUE
)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL              OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY          ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

# tinyexr v1.0.9 — single-header EXR loader; bundles miniz for zlib decoding.
# FetchContent_Populate (not MakeAvailable) downloads only — we do not process
# tinyexr's CMakeLists because the library is header-only and we define
# TINYEXR_IMPLEMENTATION ourselves in EnvMap.cpp.
FetchContent_Declare(tinyexr
    GIT_REPOSITORY https://github.com/syoyo/tinyexr.git
    GIT_TAG        v1.0.9
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(tinyexr)
if(NOT tinyexr_POPULATED)
    FetchContent_Populate(tinyexr)
endif()

# nativefiledialog-extended v1.2.1 — native OS file dialog (Open / Save)
FetchContent_Declare(nfd
    GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
    GIT_TAG        v1.2.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nfd)

# ── Project subdirectories ────────────────────────────────────────────────────
add_subdirectory(shaders)   # device_programs OBJECT library (PTX)
add_subdirectory(src)       # OptixRaytracer executable

# Set OptixRaytracer as the default startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT OptixRaytracer)
