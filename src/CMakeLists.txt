add_executable(OptixRaytracer
    main.cpp
    Application.h
    Application.cpp
    Accel.h
    Accel.cpp
    Camera.h
    Scene.h
    Scene.cpp
    Mesh.h
    Texture.h
    Texture.cpp
    SceneLoader.h
    SceneLoader.cpp
    # miniz (tinyexr's bundled zlib) — .h declares, .c defines; both are needed
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
)

target_include_directories(OptixRaytracer PRIVATE
    ${CMAKE_SOURCE_DIR}/shaders
    ${OptiX_INCLUDE_DIR}
    # tinyexr header + bundled miniz (needed by Texture.cpp which defines TINYEXR_IMPLEMENTATION)
    ${tinyexr_SOURCE_DIR}
    ${tinyexr_SOURCE_DIR}/deps/miniz
)

target_link_libraries(OptixRaytracer PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    OpenGL::GL
    glad
    glfw
    imgui
    tinygltf
    nfd
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

# Suppress LNK4098 — GLFW is built with release CRT; /NODEFAULTLIB suppresses
# the "defaultlib LIBCMT conflicts" warning in Debug builds without breaking linkage
target_link_options(OptixRaytracer PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/NODEFAULTLIB:MSVCRT>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/NODEFAULTLIB:MSVCRTD>
)

target_compile_definitions(OptixRaytracer PRIVATE NOMINMAX)

# Visual Studio source groups for cleaner Solution Explorer layout
source_group("Source Files" FILES main.cpp Application.cpp Accel.cpp Scene.cpp SceneLoader.cpp Texture.cpp)
source_group("Header Files" FILES Application.h Accel.h Camera.h Scene.h SceneLoader.h Mesh.h Texture.h)
