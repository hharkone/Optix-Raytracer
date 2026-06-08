add_executable(OptixRaytracer
    main.cpp
    Application.h
    Application.cpp
    Accel.h
    Accel.cpp
    VulkanContext.h
    VulkanContext.cpp
    Camera.h
    Matrix4x4.h
    Node3D.h
    Scene.h
    Scene.cpp
    Mesh.h
    Texture.h
    Texture.cpp
    SceneLoader.h
    SceneLoader.cpp
)

target_include_directories(OptixRaytracer PRIVATE
    ${CMAKE_SOURCE_DIR}/shaders
    ${OptiX_INCLUDE_DIR}
)

target_link_libraries(OptixRaytracer PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    Vulkan::Vulkan
    glfw
    imgui
    tinygltf
    nfd
    OpenEXR::OpenEXR
)

# Copy PTX shaders next to the executable whenever they are rebuilt.
#
# POST_BUILD on OptixRaytracer would only fire when the C++ binary relinks,
# so a shader-only change would leave a stale .ptx on disk.  Using a
# dedicated ALL custom target that depends on device_programs ensures the
# copy runs on every build where the PTX is newer — regardless of whether
# any C++ source changed.  copy_if_different makes it a no-op when the
# file is already up to date.
add_custom_target(copy_ptx ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:OptixRaytracer>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:device_programs>
            "$<TARGET_FILE_DIR:OptixRaytracer>/"
    DEPENDS device_programs
    COMMENT "Copying PTX shaders to output directory"
    COMMAND_EXPAND_LISTS
)
add_dependencies(OptixRaytracer copy_ptx)

# Install the versioned default Dear ImGui layout next to the executable the
# first time it is absent.  Uses a cmake -P script so the copy is skipped when
# the file already exists, preserving any layout the user has customised at
# runtime without requiring a manual step after every clean reconfigure.
add_custom_command(TARGET OptixRaytracer POST_BUILD
    # Alongside the executable (used when launching the .exe directly)
    COMMAND ${CMAKE_COMMAND}
        -DSRC="${CMAKE_SOURCE_DIR}/imgui.ini"
        -DDST="$<TARGET_FILE_DIR:OptixRaytracer>/imgui.ini"
        -P  "${CMAKE_SOURCE_DIR}/cmake/InstallDefaultIni.cmake"
    # In build/src/ (Visual Studio sets this as the working directory when
    # launching via F5, so ImGui looks for imgui.ini there)
    COMMAND ${CMAKE_COMMAND}
        -DSRC="${CMAKE_SOURCE_DIR}/imgui.ini"
        -DDST="${CMAKE_BINARY_DIR}/src/imgui.ini"
        -P  "${CMAKE_SOURCE_DIR}/cmake/InstallDefaultIni.cmake"
    COMMENT "Installing default imgui.ini (skipped if already present)"
)

# Suppress LNK4098 — GLFW is built with release CRT; /NODEFAULTLIB suppresses
# the "defaultlib LIBCMT conflicts" warning in Debug builds without breaking linkage
target_link_options(OptixRaytracer PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/NODEFAULTLIB:MSVCRT>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/NODEFAULTLIB:MSVCRTD>
)

target_compile_definitions(OptixRaytracer PRIVATE NOMINMAX)

# Visual Studio source groups for cleaner Solution Explorer layout
source_group("Source Files" FILES main.cpp Application.cpp VulkanContext.cpp Accel.cpp Scene.cpp SceneLoader.cpp Texture.cpp)
source_group("Header Files" FILES Application.h VulkanContext.h Accel.h Camera.h Matrix4x4.h Node3D.h Scene.h SceneLoader.h Mesh.h Texture.h)
